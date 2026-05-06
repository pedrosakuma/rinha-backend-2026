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

        var n = _norm;
        var tx = req.Transaction;
        var customer = req.Customer;
        var merchant = req.Merchant;
        var terminal = req.Terminal;

        // 0: amount / max_amount
        dst[0] = Clamp01((float)(tx.Amount / n.MaxAmount));
        // 1: installments / max_installments
        dst[1] = Clamp01(tx.Installments / n.MaxInstallments);
        // 2: (amount / customer.avg_amount) / amount_vs_avg_ratio
        var avg = (float)customer.AvgAmount;
        var ratio = avg > 0f ? (float)(tx.Amount / avg) / n.AmountVsAvgRatio : 1f;
        dst[2] = Clamp01(ratio);
        // 3: hour(requested_at) / 23  (UTC)
        var utc = tx.RequestedAt.UtcDateTime;
        dst[3] = utc.Hour / 23f;
        // 4: day_of_week / 6  (Monday=0 ... Sunday=6)
        dst[4] = MondayZero(utc.DayOfWeek) / 6f;

        // 5,6: minutes_since_last_tx / km_from_last_tx (or -1)
        if (req.LastTransaction is { } last)
        {
            var minutes = (utc - last.Timestamp.UtcDateTime).TotalMinutes;
            if (minutes < 0) minutes = 0;
            dst[5] = Clamp01((float)(minutes / n.MaxMinutes));
            dst[6] = Clamp01((float)(last.KmFromCurrent / n.MaxKm));
        }
        else
        {
            dst[5] = -1f;
            dst[6] = -1f;
        }

        // 7: km_from_home / max_km
        dst[7] = Clamp01((float)(terminal.KmFromHome / n.MaxKm));
        // 8: tx_count_24h / max_tx_count_24h
        dst[8] = Clamp01(customer.TxCount24h / n.MaxTxCount24h);
        // 9: is_online
        dst[9] = terminal.IsOnline ? 1f : 0f;
        // 10: card_present
        dst[10] = terminal.CardPresent ? 1f : 0f;
        // 11: unknown_merchant
        dst[11] = IsKnownMerchant(customer.KnownMerchants, merchant.Id) ? 0f : 1f;
        // 12: mcc_risk
        dst[12] = _mcc.Get(merchant.Mcc);
        // 13: merchant_avg_amount / max_merchant_avg_amount
        dst[13] = Clamp01((float)(merchant.AvgAmount / n.MaxMerchantAvgAmount));

        // Match oracle quantization: references in resources/references.json.gz are stored
        // pre-rounded to 4 decimal places. Round queries the same way so distance
        // comparisons (and tie-breaks) align with the oracle's k-NN ground truth.
        for (int i = 0; i < Dataset.Dimensions; i++) dst[i] = Round4dp(dst[i]);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float Round4dp(float v)
    {
        if (v < 0f) return v; // preserve -1 sentinel
        return MathF.Round(v * 10000f) / 10000f;
    }

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
