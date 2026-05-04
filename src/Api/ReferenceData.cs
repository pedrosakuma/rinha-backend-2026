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
    public const float Default = 0.5f;

    private MccRiskTable(Dictionary<string, float> values) => _values = values;

    public float Get(string mcc) => _values.TryGetValue(mcc, out var v) ? v : Default;

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
