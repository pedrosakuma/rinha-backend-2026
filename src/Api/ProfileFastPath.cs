using System.Globalization;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace Rinha.Api;

/// <summary>
/// Wave 8: bucket-keyed fast-path cache for the /fraud-score hot path.
///
/// Built once at startup from the 3M reference set: 8 of the 14 features
/// (per offline AUC analysis) are quantile-binned, producing a composite key
/// indexed into a byte table. Per-slot label is set ONLY when the bucket is
/// literally pure on training (count_fraud==0 OR count_fraud==count_total)
/// AND count >= MinBucketCount* — otherwise the slot stays "undecided" and
/// the request falls through to the scorer.
///
/// Wave 8.2: per-feature bit budget is configurable. Default keeps the original
/// uniform 3-3-3-3-3-3-3-3 (24 bits / 16 MiB), but PROFILE_FAST_PATH_BITS lets
/// the offline sweep (Bench --sweep-fastpath) propose better allocations within
/// the same memory budget.
/// </summary>
public static unsafe class ProfileFastPath
{
    // Indices into the 14-dim Vectorizer output. Wave 19: iterative greedy
    // feature swap (Bench --probe-fastpath-features) replaced amt_ratio with
    // card_present (slot 2) and tx_count_24h with is_online (slot 4),
    // lifting hit rate from 41.44% -> 74.81% on test-data, FP=FN=0.
    public static readonly int[] FeatureIndex = new[] { 0, 7, 10, 1, 9, 11, 12, 3 };
    public static readonly string[] FeatureName = new[] {
        "amount", "km_home", "card_present", "installments",
        "is_online", "unknown_merch", "mcc_risk", "hour"
    };
    public const int NumFeatures = 8;

    public const int MaxTotalBits = 24;     // 16M slots == 16 MiB byte table
    public const int MaxBitsPerFeature = 6; // 64 bins is way more than needed

    // Min-count thresholds (tunable). Asymmetric: pure-fraud needs more samples
    // (concentrated FP risk at borderline counts) than pure-legit (smaller buckets,
    // FN-safe).
    private static readonly int MinBucketCountFraud =
        int.TryParse(Environment.GetEnvironmentVariable("PROFILE_FAST_PATH_MIN_FRAUD"), out var mf) && mf > 0 ? mf : 400;
    private static readonly int MinBucketCountLegit =
        int.TryParse(Environment.GetEnvironmentVariable("PROFILE_FAST_PATH_MIN_LEGIT"), out var ml) && ml > 0 ? ml : 100;
    private static readonly int? MinBucketCountOverride =
        int.TryParse(Environment.GetEnvironmentVariable("PROFILE_FAST_PATH_MIN"), out var mo) && mo > 0 ? mo : null;

    // Hot-path arrays (set by Build).
    private static int[] _bits = Array.Empty<int>();
    private static int[] _shifts = Array.Empty<int>();
    private static float[][]? _edges;     // edges[f] length == 1 << bits[f], last = +inf
    private static byte[]? _table;        // 0=undecided, 1=legit, 2=fraud

    public const byte ResultUndecided = 0;
    public const byte ResultLegit     = 1;
    public const byte ResultFraud     = 2;

    public static bool IsEnabled => _table is not null;

    public static int UsedBuckets;
    public static int DecidedLegit;
    public static int DecidedFraud;
    public static int TotalBits;
    public static long TableSize;

    public static void Build(Dataset ds)
    {
        if (Environment.GetEnvironmentVariable("PROFILE_FAST_PATH") == "0")
        {
            Console.WriteLine("ProfileFastPath: disabled (PROFILE_FAST_PATH=0).");
            return;
        }

        var bits = ParseBits(Environment.GetEnvironmentVariable("PROFILE_FAST_PATH_BITS"));
        BuildWith(ds, bits, MinBucketCountOverride ?? MinBucketCountLegit, MinBucketCountOverride ?? MinBucketCountFraud, log: true);
    }

    /// <summary>Public Build entry point usable by Bench --sweep-fastpath.
    /// All state is global; not thread-safe across concurrent rebuilds.</summary>
    public static (int hits, int legit, int fraud, int used) BuildWith(
        Dataset ds, int[] bits, int kLegit, int kFraud, bool log)
    {
        ValidateBits(bits);
        int n = ds.Count;
        var vectors = ds.VectorsPtr;
        var labels  = ds.LabelsPtr;
        if (vectors == null || labels == null)
            throw new InvalidOperationException("Dataset missing float vectors or labels");
        const int stride = Dataset.PaddedDimensions;

        // Compute per-feature shifts and total bits.
        var shifts = new int[NumFeatures];
        int total = 0;
        for (int f = 0; f < NumFeatures; f++) { shifts[f] = total; total += bits[f]; }
        long slots = 1L << total;

        // Step 1: quantile edges (reuse single 12 MB column buffer).
        var edges = new float[NumFeatures][];
        var col = new float[n];
        for (int f = 0; f < NumFeatures; f++)
        {
            int featIdx = FeatureIndex[f];
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

        // Step 2: bucket every reference; sparse dictionary.
        var counts = new Dictionary<uint, ulong>(capacity: 65536);
        for (int i = 0; i < n; i++)
        {
            uint key = ComputeKeyStatic(vectors + (long)i * stride, bits, shifts, edges);
            ulong delta = labels[i] != 0 ? 0x1_0000_0001UL : 0x1_0000_0000UL;
            counts.TryGetValue(key, out var cur);
            counts[key] = cur + delta;
        }

        // Step 3: decide per slot. Allocate dense byte[] only at the end.
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

        // Publish.
        _bits   = bits;
        _shifts = shifts;
        _edges  = edges;
        _table  = table;
        TotalBits = total; TableSize = table.LongLength;
        UsedBuckets = used; DecidedLegit = decLegit; DecidedFraud = decFraud;

        if (log)
        {
            string bitStr = string.Join(",", bits);
            Console.WriteLine($"ProfileFastPath: built. bits=[{bitStr}] total={total} slots={slots:N0} " +
                              $"used={used:N0} decided_legit={decLegit:N0} decided_fraud={decFraud:N0} " +
                              $"k_fraud={kFraud} k_legit={kLegit} table_bytes={table.LongLength:N0}");
        }
        return (decLegit + decFraud, decLegit, decFraud, used);
    }

    public static void Disable() { _table = null; _edges = null; }

    private static int[] ParseBits(string? env)
    {
        if (string.IsNullOrWhiteSpace(env))
            // Wave 8.2: swap-greedy optimum (4 seeds → same fixed point).
            // Hits 22421/54100 (41.44%) on eval test set vs uniform 21690 (40.09%), 0 FP/FN.
            // Order matches FeatureIndex: amount, km_home, amt_ratio, installments,
            //                              tx_count_24h, unknown_merch, mcc_risk, hour.
            return new[] { 3, 3, 4, 3, 3, 4, 2, 2 };
        var parts = env.Split(',');
        if (parts.Length != NumFeatures)
            throw new ArgumentException($"PROFILE_FAST_PATH_BITS must have {NumFeatures} comma-separated ints; got '{env}'");
        var bits = new int[NumFeatures];
        for (int i = 0; i < NumFeatures; i++) bits[i] = int.Parse(parts[i].Trim(), CultureInfo.InvariantCulture);
        ValidateBits(bits);
        return bits;
    }

    private static void ValidateBits(int[] bits)
    {
        if (bits.Length != NumFeatures) throw new ArgumentException($"bits length must be {NumFeatures}");
        int total = 0;
        for (int i = 0; i < NumFeatures; i++)
        {
            if (bits[i] < 1 || bits[i] > MaxBitsPerFeature)
                throw new ArgumentException($"bits[{i}] must be in [1,{MaxBitsPerFeature}]; got {bits[i]}");
            total += bits[i];
        }
        if (total > MaxTotalBits)
            throw new ArgumentException($"sum(bits) must be <= {MaxTotalBits}; got {total}");
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static byte TryLookup(ReadOnlySpan<float> query)
    {
        var table = _table;
        if (table is null) return ResultUndecided;
        ref float q0 = ref MemoryMarshal.GetReference(query);
        var edges = _edges!;
        var bits = _bits;
        var shifts = _shifts;
        uint key = 0;
        for (int f = 0; f < NumFeatures; f++)
        {
            float v = Unsafe.Add(ref q0, FeatureIndex[f]);
            int bin = FindBin(edges[f], v);
            key |= (uint)bin << shifts[f];
        }
        return table[key];
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static uint ComputeKeyStatic(float* row, int[] bits, int[] shifts, float[][] edges)
    {
        uint key = 0;
        for (int f = 0; f < NumFeatures; f++)
        {
            float v = row[FeatureIndex[f]];
            int bin = FindBin(edges[f], v);
            key |= (uint)bin << shifts[f];
        }
        return key;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int FindBin(float[] edges, float v)
    {
        // Linear scan; edges arrays are tiny (<=64).
        for (int b = 0; b < edges.Length - 1; b++)
            if (v < edges[b]) return b;
        return edges.Length - 1;
    }
}
