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

    // Precomputed reciprocals for the hot vectorize path. Float division is ~10ns
    // and cannot be pipelined, while multiply is single-cycle on a modern x86 core.
    // The compiler won't auto-fold `/ field` into `* (1/field)` because the divisor
    // isn't a compile-time constant. ~9 divisions per request × 2 codepaths
    // (Vectorizer + JsonVectorizer) become multiplies.
    public float InvMaxAmount { get; private init; }
    public float InvMaxInstallments { get; private init; }
    public float InvAmountVsAvgRatio { get; private init; }
    public float InvMaxMinutes { get; private init; }
    public float InvMaxKm { get; private init; }
    public float InvMaxTxCount24h { get; private init; }
    public float InvMaxMerchantAvgAmount { get; private init; }

    public static NormalizationConstants Load(string path)
    {
        using var stream = File.OpenRead(path);
        using var doc = JsonDocument.Parse(stream);
        var root = doc.RootElement;
        var maxAmount = root.GetProperty("max_amount").GetSingle();
        var maxInstallments = root.GetProperty("max_installments").GetSingle();
        var amountVsAvgRatio = root.GetProperty("amount_vs_avg_ratio").GetSingle();
        var maxMinutes = root.GetProperty("max_minutes").GetSingle();
        var maxKm = root.GetProperty("max_km").GetSingle();
        var maxTxCount24h = root.GetProperty("max_tx_count_24h").GetSingle();
        var maxMerchantAvgAmount = root.GetProperty("max_merchant_avg_amount").GetSingle();
        return new NormalizationConstants
        {
            MaxAmount = maxAmount,
            MaxInstallments = maxInstallments,
            AmountVsAvgRatio = amountVsAvgRatio,
            MaxMinutes = maxMinutes,
            MaxKm = maxKm,
            MaxTxCount24h = maxTxCount24h,
            MaxMerchantAvgAmount = maxMerchantAvgAmount,
            InvMaxAmount = 1f / maxAmount,
            InvMaxInstallments = 1f / maxInstallments,
            InvAmountVsAvgRatio = 1f / amountVsAvgRatio,
            InvMaxMinutes = 1f / maxMinutes,
            InvMaxKm = 1f / maxKm,
            InvMaxTxCount24h = 1f / maxTxCount24h,
            InvMaxMerchantAvgAmount = 1f / maxMerchantAvgAmount,
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
    // Q16 mirror of _valuesPacked for the int16 vectorize path: each entry is
    // (short)Round(v * Dataset.Q16Scale). Lets the brute-Q16 hot path skip the
    // round+cast on every request.
    private readonly short[] _valuesPackedQ16;
    public const float Default = 0.5f;
    public const short DefaultQ16 = 5000; // Round(0.5f * 10000) = 5000

    private MccRiskTable(Dictionary<string, float> values)
    {
        _values = values;
        _keysPacked = new uint[values.Count];
        _valuesPacked = new float[values.Count];
        _valuesPackedQ16 = new short[values.Count];
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
            int q = (int)MathF.Round(kv.Value * Dataset.Q16Scale);
            if (q > 32767) q = 32767;
            else if (q < -32768) q = -32768;
            _valuesPackedQ16[i] = (short)q;
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

    /// <summary>Q16 version of <see cref="Get(string)"/>: returns the pre-quantized
    /// (short)Round(v * Q16Scale) value. Saves a Round+cast per request on the brute-Q16 path.</summary>
    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
    public short GetQ16(string mcc)
    {
        if (mcc is null || mcc.Length != 4) return DefaultQ16;
        uint packed = (uint)mcc[0]
                    | ((uint)mcc[1] << 8)
                    | ((uint)mcc[2] << 16)
                    | ((uint)mcc[3] << 24);
        var keys = _keysPacked;
        var values = _valuesPackedQ16;
        for (int i = 0; i < keys.Length; i++)
        {
            if (keys[i] == packed) return values[i];
        }
        return DefaultQ16;
    }

    /// <summary>Q16 version of <see cref="Get(ReadOnlySpan{byte})"/>: zero-alloc lookup
    /// returning the pre-quantized short.</summary>
    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
    public short GetQ16(ReadOnlySpan<byte> mccBytes)
    {
        if (mccBytes.Length != 4) return DefaultQ16;
        uint packed = System.Buffers.Binary.BinaryPrimitives.ReadUInt32LittleEndian(mccBytes);
        var keys = _keysPacked;
        var values = _valuesPackedQ16;
        for (int i = 0; i < keys.Length; i++)
        {
            if (keys[i] == packed) return values[i];
        }
        return DefaultQ16;
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
