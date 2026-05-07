using System.Text.Json;

namespace Rinha.Api;

public sealed class NormalizationConstants
{
    public required float MaxAmount { get; init; }
    public required float MaxInstallments { get; init; }
    public required float AmountVsAvgRatio { get; init; }
    public required float MaxMinutes { get; init; }
    public required float MaxKm { get; init; }
    public required float MaxTxCount24h { get; init; }
    public required float MaxMerchantAvgAmount { get; init; }

    public static NormalizationConstants Load(string path)
    {
        using var stream = File.OpenRead(path);
        using var doc = JsonDocument.Parse(stream);
        var root = doc.RootElement;
        return new NormalizationConstants
        {
            MaxAmount = root.GetProperty("max_amount").GetSingle(),
            MaxInstallments = root.GetProperty("max_installments").GetSingle(),
            AmountVsAvgRatio = root.GetProperty("amount_vs_avg_ratio").GetSingle(),
            MaxMinutes = root.GetProperty("max_minutes").GetSingle(),
            MaxKm = root.GetProperty("max_km").GetSingle(),
            MaxTxCount24h = root.GetProperty("max_tx_count_24h").GetSingle(),
            MaxMerchantAvgAmount = root.GetProperty("max_merchant_avg_amount").GetSingle(),
        };
    }
}

public sealed class MccRiskTable
{
    private readonly Dictionary<string, float> _values;
    // Parallel byte-keyed lookup for the JSON hot path (zero-alloc).
    // MCC keys ship as exactly 4 ASCII digits in the dataset, so we pack each
    // into a uint32 and linear-scan the small table (≤16 entries fits one cache line).
    private readonly uint[] _keysPacked;
    private readonly float[] _valuesPacked;
    public const float Default = 0.5f;

    private MccRiskTable(Dictionary<string, float> values)
    {
        _values = values;
        _keysPacked = new uint[values.Count];
        _valuesPacked = new float[values.Count];
        int i = 0;
        foreach (var kv in values)
        {
            if (kv.Key.Length != 4)
                throw new InvalidDataException($"MCC key '{kv.Key}' is not 4 chars (packed-uint lookup invariant)");
            uint packed = (uint)kv.Key[0]
                        | ((uint)kv.Key[1] << 8)
                        | ((uint)kv.Key[2] << 16)
                        | ((uint)kv.Key[3] << 24);
            _keysPacked[i] = packed;
            _valuesPacked[i] = kv.Value;
            i++;
        }
    }

    public float Get(string mcc) => _values.TryGetValue(mcc, out var v) ? v : Default;

    /// <summary>Zero-alloc lookup keyed on the raw 4-byte ASCII MCC span (e.g., taken
    /// directly from Utf8JsonReader.ValueSpan). Returns Default for any non-4-byte input.</summary>
    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
    public float Get(ReadOnlySpan<byte> mccBytes)
    {
        if (mccBytes.Length != 4) return Default;
        uint packed = System.Buffers.Binary.BinaryPrimitives.ReadUInt32LittleEndian(mccBytes);
        var keys = _keysPacked;
        var values = _valuesPacked;
        for (int i = 0; i < keys.Length; i++)
        {
            if (keys[i] == packed) return values[i];
        }
        return Default;
    }

    public static MccRiskTable Load(string path)
    {
        using var stream = File.OpenRead(path);
        using var doc = JsonDocument.Parse(stream);
        var dict = new Dictionary<string, float>(StringComparer.Ordinal);
        foreach (var prop in doc.RootElement.EnumerateObject())
        {
            dict[prop.Name] = prop.Value.GetSingle();
        }
        return new MccRiskTable(dict);
    }
}
