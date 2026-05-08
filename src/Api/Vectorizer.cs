using System.Runtime.CompilerServices;

namespace Rinha.Api;

public sealed class Vectorizer
{
    private readonly NormalizationConstants _norm;
    private readonly MccRiskTable _mcc;

    public Vectorizer(NormalizationConstants norm, MccRiskTable mcc)
    {
        _norm = norm;
        _mcc = mcc;
    }

    public void Vectorize(FraudRequest req, Span<float> dst)
    {
        if (dst.Length < Dataset.Dimensions)
            throw new ArgumentException("Destination too small", nameof(dst));

        Span<short> q = stackalloc short[Dataset.Dimensions];
        VectorizeCore(req, q);
        for (int i = 0; i < Dataset.Dimensions; i++) dst[i] = q[i] * (1f / Dataset.Q16Scale);
    }

    /// <summary>Q16 path: produces the canonical int16 query directly, no float
    /// intermediate. Each entry is <c>Round(feature * Dataset.Q16Scale)</c>, i.e.
    /// what the brute-Q16 scorer expects after its own round+cast — so the scorer
    /// can skip that step when fed by this method.</summary>
    public void VectorizeQ16(FraudRequest req, Span<short> dst)
    {
        if (dst.Length < Dataset.Dimensions)
            throw new ArgumentException("Destination too small", nameof(dst));
        VectorizeCore(req, dst);
    }

    /// <summary>Combined: produces both float and Q16 representations in one pass.
    /// The float values are bit-equal to <see cref="Vectorize"/> alone since they
    /// are derived from the canonical short via <c>q / Q16Scale</c>.</summary>
    public void Vectorize(FraudRequest req, Span<float> floatDst, Span<short> q16Dst)
    {
        if (floatDst.Length < Dataset.Dimensions || q16Dst.Length < Dataset.Dimensions)
            throw new ArgumentException("Destination too small");
        VectorizeCore(req, q16Dst);
        for (int i = 0; i < Dataset.Dimensions; i++) floatDst[i] = q16Dst[i] * (1f / Dataset.Q16Scale);
    }

    private void VectorizeCore(FraudRequest req, Span<short> dst)
    {
        var n = _norm;
        var tx = req.Transaction;
        var customer = req.Customer;
        var merchant = req.Merchant;
        var terminal = req.Terminal;

        // 0: amount / max_amount
        dst[0] = Q16(Clamp01((float)(tx.Amount * n.InvMaxAmount)));
        // 1: installments / max_installments
        dst[1] = Q16(Clamp01(tx.Installments * n.InvMaxInstallments));
        // 2: (amount / customer.avg_amount) / amount_vs_avg_ratio
        var avg = (float)customer.AvgAmount;
        var ratio = avg > 0f ? (float)(tx.Amount / avg) * n.InvAmountVsAvgRatio : 1f;
        dst[2] = Q16(Clamp01(ratio));
        // 3: hour(requested_at) / 23  (UTC)
        var utc = tx.RequestedAt.UtcDateTime;
        dst[3] = Q16(utc.Hour * (1f / 23f));
        // 4: day_of_week / 6  (Monday=0 ... Sunday=6)
        dst[4] = Q16(MondayZero(utc.DayOfWeek) * (1f / 6f));

        // 5,6: minutes_since_last_tx / km_from_last_tx (or -1)
        if (req.LastTransaction is { } last)
        {
            var minutes = (utc - last.Timestamp.UtcDateTime).TotalMinutes;
            if (minutes < 0) minutes = 0;
            dst[5] = Q16(Clamp01((float)(minutes * n.InvMaxMinutes)));
            dst[6] = Q16(Clamp01((float)(last.KmFromCurrent * n.InvMaxKm)));
        }
        else
        {
            dst[5] = -10000;
            dst[6] = -10000;
        }

        // 7: km_from_home / max_km
        dst[7] = Q16(Clamp01((float)(terminal.KmFromHome * n.InvMaxKm)));
        // 8: tx_count_24h / max_tx_count_24h
        dst[8] = Q16(Clamp01(customer.TxCount24h * n.InvMaxTxCount24h));
        // 9-11: boolean features — exactly 0 or 10000 in Q16 units.
        dst[9] = (short)(terminal.IsOnline ? 10000 : 0);
        dst[10] = (short)(terminal.CardPresent ? 10000 : 0);
        dst[11] = (short)(IsKnownMerchant(customer.KnownMerchants, merchant.Id) ? 0 : 10000);
        // 12: mcc_risk (table values pre-quantized at load time)
        dst[12] = _mcc.GetQ16(merchant.Mcc);
        // 13: merchant_avg_amount / max_merchant_avg_amount
        dst[13] = Q16(Clamp01((float)(merchant.AvgAmount * n.InvMaxMerchantAvgAmount)));
    }

    /// <summary>Single canonical quantization: <c>(short)Round(v * Q16Scale)</c>.
    /// Inputs are already in [0,1] (or exactly -1 for sentinels handled at the
    /// caller), so no clamping needed here. Round-to-nearest-even matches the
    /// reference data preprocessor (4dp quantization).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static short Q16(float v) => (short)MathF.Round(v * Dataset.Q16Scale);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float Clamp01(float v) => v < 0f ? 0f : (v > 1f ? 1f : v);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int MondayZero(DayOfWeek d) => d == DayOfWeek.Sunday ? 6 : (int)d - 1;

    private static bool IsKnownMerchant(List<string>? known, string merchantId)
    {
        if (known is null || known.Count == 0) return false;
        for (int i = 0; i < known.Count; i++)
        {
            if (string.Equals(known[i], merchantId, StringComparison.Ordinal))
                return true;
        }
        return false;
    }
}
