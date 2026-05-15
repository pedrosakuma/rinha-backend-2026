using System.Globalization;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text.Json;

namespace Rinha.Api;

/// <summary>
/// Cascade of selective decision tables with abstention. Each stage returns a
/// fraud_count in [0,5] only for buckets it considers safe; otherwise it returns
/// Undecided and the next stage or full scorer handles the request.
/// </summary>
public sealed unsafe class SelectiveDecisionCascade
{
    public const byte ResultUndecided = 255;

    private readonly SelectiveDecisionTable[] _stages;

    private SelectiveDecisionCascade(SelectiveDecisionTable[] stages)
        => _stages = stages;

    public bool IsEnabled => _stages.Length != 0;
    public IReadOnlyList<SelectiveDecisionTable> Stages => _stages;

    public static SelectiveDecisionCascade Empty { get; } = new(Array.Empty<SelectiveDecisionTable>());

    public static SelectiveDecisionCascade Build(Dataset ds, string configPath)
    {
        if (!File.Exists(configPath))
        {
            Console.WriteLine($"SelectiveDecisionCascade: config not found at {configPath}; cascade disabled.");
            return Empty;
        }

        var config = SelectiveDecisionConfig.Load(configPath);
        var stages = new List<SelectiveDecisionTable>(config.Stages.Length);
        foreach (var stage in config.Stages)
        {
            if (!stage.IsEnabled())
            {
                Console.WriteLine($"SelectiveDecisionTable[{stage.Name}]: disabled.");
                continue;
            }

            stages.Add(stage.Mode switch
            {
                "reference_purity" => SelectiveDecisionTable.BuildReferencePurity(ds, stage),
                "residual_modal_sparse" => SelectiveDecisionTable.BuildSparse(ds, stage),
                _ => throw new InvalidOperationException($"Unknown selective decision table mode '{stage.Mode}' for stage '{stage.Name}'")
            });
        }

        Console.WriteLine($"SelectiveDecisionCascade: enabled stages={stages.Count}.");
        return stages.Count == 0 ? Empty : new SelectiveDecisionCascade(stages.ToArray());
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public byte TryLookup(ReadOnlySpan<float> query)
    {
        var stages = _stages;
        for (int i = 0; i < stages.Length; i++)
        {
            byte result = stages[i].TryLookup(query);
            if (result != ResultUndecided) return result;
        }
        return ResultUndecided;
    }

    public int StageCount => _stages.Length;

    public byte TryLookupWithStage(ReadOnlySpan<float> query, out int stageIndex)
    {
        var stages = _stages;
        for (int i = 0; i < stages.Length; i++)
        {
            byte result = stages[i].TryLookup(query);
            if (result != ResultUndecided)
            {
                stageIndex = i;
                return result;
            }
        }
        stageIndex = -1;
        return ResultUndecided;
    }
}

public sealed unsafe class SelectiveDecisionTable
{
    public const int MaxTotalBits = 24;
    public const int MaxBitsPerFeature = 12;

    private readonly int[] _featureIndices;
    private readonly int[] _bits;
    private readonly int[] _shifts;
    private readonly float[][] _edges;
    private readonly byte[] _table;
    private readonly int _numFeatures;

    private SelectiveDecisionTable(
        SelectiveDecisionStageConfig config,
        int[] shifts,
        float[][] edges,
        byte[] table,
        int usedBuckets,
        int[] decidedByCount)
    {
        Name = config.Name;
        Mode = config.Mode;
        RiskLevel = config.RiskLevel;
        Source = config.Source;
        EnabledByDefault = config.EnabledByDefault;
        EnvFlags = config.EnvFlags;
        _featureIndices = config.FeatureIndices;
        _bits = config.Bits;
        _shifts = shifts;
        _edges = edges;
        _table = table;
        _numFeatures = config.FeatureIndices.Length;
        UsedBuckets = usedBuckets;
        DecidedByCount = decidedByCount;
        TotalBits = config.Bits.Sum();
        TableSize = table.LongLength;
    }

    public string Name { get; }
    public string Mode { get; }
    public string RiskLevel { get; }
    public string Source { get; }
    public bool EnabledByDefault { get; }
    public string[] EnvFlags { get; }
    public int UsedBuckets { get; }
    public int[] DecidedByCount { get; }
    public int TotalBits { get; }
    public long TableSize { get; }

    public static SelectiveDecisionTable BuildReferencePurity(Dataset ds, SelectiveDecisionStageConfig config)
    {
        config = ApplyReferencePurityLegacyOverrides(config);
        ValidateFeatureConfig(config.FeatureIndices, config.Bits);
        var (shifts, totalBits) = BuildShifts(config.Bits);
        long slots = 1L << totalBits;
        var edges = BuildEdges(ds, config.FeatureIndices, config.Bits);
        var counts = CountReferenceLabels(ds, config.FeatureIndices, config.Bits, shifts, edges);

        var table = NewUndecidedTable(slots);
        var decided = new int[6];
        foreach (var kv in counts)
        {
            int total = (int)(kv.Value >> 32);
            int positives = (int)(uint)kv.Value;
            if (positives == 0 && total >= config.KLegit)
            {
                table[kv.Key] = 0;
                decided[0]++;
            }
            else if (positives == total && total >= config.KFraud)
            {
                table[kv.Key] = 5;
                decided[5]++;
            }
        }

        var built = new SelectiveDecisionTable(config, shifts, edges, table, counts.Count, decided);
        built.LogBuilt();
        return built;
    }

    private static SelectiveDecisionStageConfig ApplyReferencePurityLegacyOverrides(SelectiveDecisionStageConfig config)
    {
        if (config.Name != "reference-purity-1")
            return config;

        int[] bits = ParseBitsOverride(Environment.GetEnvironmentVariable("PROFILE_FAST_PATH_BITS"), config.Bits);
        int? minOverride = ParsePositiveInt(Environment.GetEnvironmentVariable("PROFILE_FAST_PATH_MIN"));
        int kLegit = minOverride
            ?? ParsePositiveInt(Environment.GetEnvironmentVariable("PROFILE_FAST_PATH_MIN_LEGIT"))
            ?? config.KLegit;
        int kFraud = minOverride
            ?? ParsePositiveInt(Environment.GetEnvironmentVariable("PROFILE_FAST_PATH_MIN_FRAUD"))
            ?? config.KFraud;
        return config with { Bits = bits, KLegit = kLegit, KFraud = kFraud };
    }

    private static int[] ParseBitsOverride(string? env, int[] fallback)
    {
        if (string.IsNullOrWhiteSpace(env)) return fallback;
        var parts = env.Split(',');
        if (parts.Length != fallback.Length)
            throw new ArgumentException($"PROFILE_FAST_PATH_BITS must have {fallback.Length} comma-separated ints; got '{env}'");
        var bits = new int[parts.Length];
        for (int i = 0; i < parts.Length; i++)
            bits[i] = int.Parse(parts[i].Trim(), CultureInfo.InvariantCulture);
        return bits;
    }

    private static int? ParsePositiveInt(string? value)
        => int.TryParse(value, NumberStyles.Integer, CultureInfo.InvariantCulture, out int parsed) && parsed > 0 ? parsed : null;

    public static SelectiveDecisionTable BuildSparse(Dataset ds, SelectiveDecisionStageConfig config)
    {
        ValidateFeatureConfig(config.FeatureIndices, config.Bits);
        if (config.CountKeys.Length != 6)
            throw new InvalidOperationException($"Stage '{config.Name}' must provide 6 count key arrays");

        var (shifts, totalBits) = BuildShifts(config.Bits);
        long slots = 1L << totalBits;
        var edges = BuildEdges(ds, config.FeatureIndices, config.Bits);
        var table = NewUndecidedTable(slots);
        var decided = new int[6];

        for (int count = 0; count <= 5; count++)
        {
            foreach (uint key in config.CountKeys[count])
            {
                if (key >= slots)
                    throw new InvalidOperationException($"Stage '{config.Name}' count{count} key {key} outside table size {slots}");
                if (table[key] == SelectiveDecisionCascade.ResultUndecided)
                {
                    table[key] = (byte)count;
                    decided[count]++;
                }
                else if (table[key] != count)
                {
                    throw new InvalidOperationException($"Stage '{config.Name}' duplicate key {key} with conflicting result");
                }
            }
        }

        var built = new SelectiveDecisionTable(config, shifts, edges, table, decided.Sum(), decided);
        built.LogBuilt();
        return built;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public byte TryLookup(ReadOnlySpan<float> query)
    {
        ref float q0 = ref MemoryMarshal.GetReference(query);
        uint key = 0;
        for (int f = 0; f < _numFeatures; f++)
        {
            float v = Unsafe.Add(ref q0, _featureIndices[f]);
            int bin = FindBin(_edges[f], v);
            key |= (uint)bin << _shifts[f];
        }
        return _table[key];
    }

    private void LogBuilt()
    {
        Console.WriteLine($"SelectiveDecisionTable[{Name}]: built. mode={Mode} risk={RiskLevel} " +
                          $"features=[{string.Join(",", _featureIndices)}] bits=[{string.Join(",", _bits)}] " +
                          $"total={TotalBits} slots={_table.LongLength:N0} used={UsedBuckets:N0} " +
                          $"decided_counts=[{string.Join(",", DecidedByCount)}] table_bytes={_table.LongLength:N0}");
    }

    private static byte[] NewUndecidedTable(long slots)
    {
        if (slots > int.MaxValue)
            throw new InvalidOperationException($"Selective decision table too large: {slots} slots");
        var table = new byte[slots];
        Array.Fill(table, SelectiveDecisionCascade.ResultUndecided);
        return table;
    }

    private static (int[] shifts, int totalBits) BuildShifts(int[] bits)
    {
        var shifts = new int[bits.Length];
        int total = 0;
        for (int i = 0; i < bits.Length; i++)
        {
            shifts[i] = total;
            total += bits[i];
        }
        return (shifts, total);
    }

    private static void ValidateFeatureConfig(int[] featureIndices, int[] bits)
    {
        if (featureIndices.Length != bits.Length)
            throw new ArgumentException("feature_indices length must equal bits length");
        if (featureIndices.Length == 0)
            throw new ArgumentException("selective decision table must include at least one feature");
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

    private static float[][] BuildEdges(Dataset ds, int[] featureIndices, int[] bits)
    {
        int n = ds.Count;
        var vectors = ds.VectorsPtr;
        if (vectors == null)
            throw new InvalidOperationException("Dataset missing float vectors");
        const int stride = Dataset.PaddedDimensions;

        var edges = new float[featureIndices.Length][];
        var col = new float[n];
        for (int f = 0; f < featureIndices.Length; f++)
        {
            int featureIdx = featureIndices[f];
            for (int i = 0; i < n; i++)
                col[i] = vectors[(long)i * stride + featureIdx];
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
        return edges;
    }

    private static Dictionary<uint, ulong> CountReferenceLabels(Dataset ds, int[] featureIndices, int[] bits, int[] shifts, float[][] edges)
    {
        int n = ds.Count;
        var vectors = ds.VectorsPtr;
        var labels = ds.LabelsPtr;
        if (vectors == null || labels == null)
            throw new InvalidOperationException("Dataset missing float vectors or labels");
        const int stride = Dataset.PaddedDimensions;

        var counts = new Dictionary<uint, ulong>(capacity: 65536);
        for (int i = 0; i < n; i++)
        {
            uint key = ComputeKeyStatic(vectors + (long)i * stride, featureIndices, shifts, edges);
            ulong delta = labels[i] != 0 ? 0x1_0000_0001UL : 0x1_0000_0000UL;
            counts.TryGetValue(key, out var cur);
            counts[key] = cur + delta;
        }
        return counts;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static uint ComputeKeyStatic(float* row, int[] featureIndices, int[] shifts, float[][] edges)
    {
        uint key = 0;
        for (int f = 0; f < featureIndices.Length; f++)
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
}

public sealed record SelectiveDecisionConfig(SelectiveDecisionStageConfig[] Stages)
{
    public static SelectiveDecisionConfig Load(string path)
    {
        using var doc = JsonDocument.Parse(File.ReadAllBytes(path));
        var root = doc.RootElement;
        if (!root.TryGetProperty("stages", out var stagesEl) || stagesEl.ValueKind != JsonValueKind.Array)
            throw new InvalidOperationException($"{Path.GetFileName(path)} missing array 'stages'");

        var stages = new List<SelectiveDecisionStageConfig>(stagesEl.GetArrayLength());
        foreach (var stageEl in stagesEl.EnumerateArray())
            stages.Add(SelectiveDecisionStageConfig.Parse(stageEl));
        return new SelectiveDecisionConfig(stages.ToArray());
    }
}

public sealed record SelectiveDecisionStageConfig(
    string Name,
    string Mode,
    bool EnabledByDefault,
    string[] EnvFlags,
    string RiskLevel,
    string Source,
    int[] FeatureIndices,
    string[] FeatureNames,
    int[] Bits,
    int KLegit,
    int KFraud,
    int MinQuerySupport,
    uint[][] CountKeys)
{
    public bool IsEnabled()
    {
        bool sawExplicit = false;
        bool enabled = EnabledByDefault;
        foreach (string flag in EnvFlags)
        {
            string? value = Environment.GetEnvironmentVariable(flag);
            if (string.IsNullOrWhiteSpace(value)) continue;
            sawExplicit = true;
            if (value == "1") enabled = true;
            else if (value == "0") enabled = false;
        }
        return EnabledByDefault ? enabled : sawExplicit && enabled;
    }

    public static SelectiveDecisionStageConfig Parse(JsonElement el)
    {
        string name = GetString(el, "name");
        string mode = GetString(el, "mode");
        bool enabledByDefault = el.TryGetProperty("enabled_by_default", out var enabledEl) && enabledEl.GetBoolean();
        string[] envFlags = el.TryGetProperty("env_flags", out var envEl) ? ParseStringArray(envEl) : Array.Empty<string>();
        string riskLevel = el.TryGetProperty("risk_level", out var riskEl) ? riskEl.GetString() ?? "unknown" : "unknown";
        string source = el.TryGetProperty("source", out var sourceEl) ? sourceEl.GetString() ?? "" : "";
        int[] featureIndices = ParseIntArray(el, "feature_indices");
        string[] featureNames = el.TryGetProperty("feature_names", out var featureNamesEl) ? ParseStringArray(featureNamesEl) : Array.Empty<string>();
        int[] bits = ParseIntArray(el, "bits");
        int kLegit = el.TryGetProperty("k_legit", out var kl) ? kl.GetInt32() : 0;
        int kFraud = el.TryGetProperty("k_fraud", out var kf) ? kf.GetInt32() : 0;
        int minQuerySupport = el.TryGetProperty("min_query_support", out var mqs) ? mqs.GetInt32() : 0;
        var countKeys = new uint[6][];
        for (int count = 0; count <= 5; count++)
        {
            countKeys[count] = el.TryGetProperty($"count{count}_keys", out var keysEl)
                ? ParseUIntArray(keysEl)
                : Array.Empty<uint>();
        }

        return new SelectiveDecisionStageConfig(
            name, mode, enabledByDefault, envFlags, riskLevel, source,
            featureIndices, featureNames, bits, kLegit, kFraud, minQuerySupport, countKeys);
    }

    private static string GetString(JsonElement root, string name)
    {
        if (!root.TryGetProperty(name, out var el) || el.ValueKind != JsonValueKind.String)
            throw new InvalidOperationException($"selective decision stage missing string '{name}'");
        return el.GetString()!;
    }

    private static int[] ParseIntArray(JsonElement root, string name)
    {
        if (!root.TryGetProperty(name, out var el) || el.ValueKind != JsonValueKind.Array)
            throw new InvalidOperationException($"selective decision stage missing array '{name}'");
        var values = new int[el.GetArrayLength()];
        int i = 0;
        foreach (var v in el.EnumerateArray())
            values[i++] = v.GetInt32();
        return values;
    }

    private static uint[] ParseUIntArray(JsonElement el)
    {
        if (el.ValueKind != JsonValueKind.Array)
            throw new InvalidOperationException("selective decision key list must be an array");
        var values = new uint[el.GetArrayLength()];
        int i = 0;
        foreach (var v in el.EnumerateArray())
            values[i++] = v.GetUInt32();
        return values;
    }

    private static string[] ParseStringArray(JsonElement el)
    {
        if (el.ValueKind != JsonValueKind.Array)
            throw new InvalidOperationException("selective decision string list must be an array");
        var values = new string[el.GetArrayLength()];
        int i = 0;
        foreach (var v in el.EnumerateArray())
            values[i++] = v.GetString() ?? "";
        return values;
    }
}
