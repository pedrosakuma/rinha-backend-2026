using System.Globalization;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text.Json;

namespace Rinha.Api;

/// <summary>
/// Wave 29: 2nd-level bucket fast-path, runs when <see cref="ProfileFastPath"/>
/// returns Undecided. Same architecture (quantile-binned bucket table over a
/// subset of features), but uses a *different* feature mix tuned to attack the
/// residue that FP1 cannot decide. This is what eats 96% of the scorer CPU.
///
/// All knobs (feature indices, per-feature bits, k_fraud, k_legit) come from
/// <c>resources/profile_fastpath2.json</c> — produced by the offline tool
/// <c>Bench --build-fastpath2</c>. No thresholds are hardcoded in C#.
///
/// The lookup table is built at startup from <c>references.bin</c>/<c>labels.bin</c>
/// using identical voting logic to FP1: a bucket only decides Legit/Fraud when it
/// is *pure* on training and the support count meets the per-side threshold.
/// </summary>
public static unsafe class ProfileFastPath2
{
    public const byte ResultUndecided = ProfileFastPath.ResultUndecided;
    public const byte ResultLegit     = ProfileFastPath.ResultLegit;
    public const byte ResultFraud     = ProfileFastPath.ResultFraud;

    public const int MaxTotalBits = 24;
    public const int MaxBitsPerFeature = 6;

    private static int[] _featureIndices = Array.Empty<int>();
    private static int[] _bits   = Array.Empty<int>();
    private static int[] _shifts = Array.Empty<int>();
    private static float[][]? _edges;
    private static byte[]?    _table;
    private static int        _numFeatures;

    public static bool IsEnabled => _table is not null;
    public static int  UsedBuckets;
    public static int  DecidedLegit;
    public static int  DecidedFraud;
    public static int  TotalBits;
    public static long TableSize;

    /// <summary>
    /// Loads the config JSON and builds the lookup table from references.
    /// Disabled (no-op) if PROFILE_FAST_PATH2=0 or the config file is missing.
    /// </summary>
    public static void Build(Dataset ds, string configPath)
    {
        if (Environment.GetEnvironmentVariable("PROFILE_FAST_PATH2") == "0")
        {
            Console.WriteLine("ProfileFastPath2: disabled (PROFILE_FAST_PATH2=0).");
            return;
        }
        if (!File.Exists(configPath))
        {
            Console.WriteLine($"ProfileFastPath2: config not found at {configPath}; FP2 disabled.");
            return;
        }
        var cfg = LoadConfig(configPath);
        BuildWith(ds, cfg.FeatureIndices, cfg.Bits, cfg.KLegit, cfg.KFraud, log: true);
    }

    public static (int hits, int legit, int fraud, int used) BuildWith(
        Dataset ds, int[] featureIndices, int[] bits, int kLegit, int kFraud, bool log)
    {
        if (featureIndices.Length != bits.Length)
            throw new ArgumentException("featureIndices.Length must equal bits.Length");
        ValidateBits(bits);

        int nf = featureIndices.Length;
        int n  = ds.Count;
        var vectors = ds.VectorsPtr;
        var labels  = ds.LabelsPtr;
        if (vectors == null || labels == null)
            throw new InvalidOperationException("Dataset missing float vectors or labels");
        const int stride = Dataset.PaddedDimensions;

        var shifts = new int[nf];
        int total = 0;
        for (int f = 0; f < nf; f++) { shifts[f] = total; total += bits[f]; }
        long slots = 1L << total;

        // Quantile edges per feature.
        var edges = new float[nf][];
        var col = new float[n];
        for (int f = 0; f < nf; f++)
        {
            int featIdx = featureIndices[f];
            for (int i = 0; i < n; i++) col[i] = vectors[(long)i * stride + featIdx];
            Array.Sort(col);
            int binCount = 1 << bits[f];
            edges[f] = new float[binCount];
            for (int b = 0; b < binCount - 1; b++)
            {
                int q = (int)((long)(b + 1) * n / binCount);
                edges[f][b] = col[q];
            }
            edges[f][binCount - 1] = float.PositiveInfinity;
        }
        col = null!;

        // Bucket every reference. Sparse dictionary; high bits of value = total, low = positives.
        var counts = new Dictionary<uint, ulong>(capacity: 65536);
        for (int i = 0; i < n; i++)
        {
            uint key = ComputeKeyStatic(vectors + (long)i * stride, featureIndices, bits, shifts, edges);
            ulong delta = labels[i] != 0 ? 0x1_0000_0001UL : 0x1_0000_0000UL;
            counts.TryGetValue(key, out var cur);
            counts[key] = cur + delta;
        }

        var table = new byte[slots];
        int used = counts.Count, decLegit = 0, decFraud = 0;
        foreach (var kv in counts)
        {
            int t = (int)(kv.Value >> 32);
            int p = (int)(uint)kv.Value;
            if (p == 0 && t >= kLegit)      { table[kv.Key] = ResultLegit; decLegit++; }
            else if (p == t && t >= kFraud) { table[kv.Key] = ResultFraud; decFraud++; }
        }
        counts = null!;

        _featureIndices = featureIndices;
        _bits           = bits;
        _shifts         = shifts;
        _edges          = edges;
        _table          = table;
        _numFeatures    = nf;
        TotalBits       = total;
        TableSize       = table.LongLength;
        UsedBuckets     = used;
        DecidedLegit    = decLegit;
        DecidedFraud    = decFraud;

        if (log)
        {
            string bitStr  = string.Join(",", bits);
            string featStr = string.Join(",", featureIndices);
            Console.WriteLine($"ProfileFastPath2: built. features=[{featStr}] bits=[{bitStr}] " +
                              $"total={total} slots={slots:N0} used={used:N0} " +
                              $"decided_legit={decLegit:N0} decided_fraud={decFraud:N0} " +
                              $"k_fraud={kFraud} k_legit={kLegit} table_bytes={table.LongLength:N0}");
        }
        return (decLegit + decFraud, decLegit, decFraud, used);
    }

    public static void Disable() { _table = null; _edges = null; }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static byte TryLookup(ReadOnlySpan<float> query)
    {
        var table = _table;
        if (table is null) return ResultUndecided;
        ref float q0 = ref MemoryMarshal.GetReference(query);
        var edges    = _edges!;
        var shifts   = _shifts;
        var feats    = _featureIndices;
        int nf       = _numFeatures;
        uint key = 0;
        for (int f = 0; f < nf; f++)
        {
            float v = Unsafe.Add(ref q0, feats[f]);
            int bin = FindBin(edges[f], v);
            key |= (uint)bin << shifts[f];
        }
        return table[key];
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static uint ComputeKeyStatic(float* row, int[] featureIndices, int[] bits, int[] shifts, float[][] edges)
    {
        uint key = 0;
        int nf = featureIndices.Length;
        for (int f = 0; f < nf; f++)
        {
            float v = row[featureIndices[f]];
            int bin = FindBin(edges[f], v);
            key |= (uint)bin << shifts[f];
        }
        return key;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int FindBin(float[] edges, float v)
    {
        for (int b = 0; b < edges.Length - 1; b++)
            if (v < edges[b]) return b;
        return edges.Length - 1;
    }

    private static void ValidateBits(int[] bits)
    {
        if (bits.Length < 1) throw new ArgumentException("bits must have at least one entry");
        int total = 0;
        for (int i = 0; i < bits.Length; i++)
        {
            if (bits[i] < 1 || bits[i] > MaxBitsPerFeature)
                throw new ArgumentException($"bits[{i}] must be in [1,{MaxBitsPerFeature}]; got {bits[i]}");
            total += bits[i];
        }
        if (total > MaxTotalBits)
            throw new ArgumentException($"sum(bits) must be <= {MaxTotalBits}; got {total}");
    }

    public sealed record Config(int[] FeatureIndices, int[] Bits, int KFraud, int KLegit);

    public static Config LoadConfig(string path)
    {
        var bytes = File.ReadAllBytes(path);
        using var doc = JsonDocument.Parse(bytes);
        var root = doc.RootElement;
        var feats = ParseIntArray(root, "feature_indices");
        var bits  = ParseIntArray(root, "bits");
        int kFraud = root.TryGetProperty("k_fraud", out var kf) ? kf.GetInt32() : 200;
        int kLegit = root.TryGetProperty("k_legit", out var kl) ? kl.GetInt32() : 50;
        return new Config(feats, bits, kFraud, kLegit);
    }

    private static int[] ParseIntArray(JsonElement root, string name)
    {
        if (!root.TryGetProperty(name, out var el) || el.ValueKind != JsonValueKind.Array)
            throw new InvalidOperationException($"profile_fastpath2.json missing array '{name}'");
        var arr = new int[el.GetArrayLength()];
        int i = 0;
        foreach (var v in el.EnumerateArray()) arr[i++] = v.GetInt32();
        return arr;
    }
}
