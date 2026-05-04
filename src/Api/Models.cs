using System.Text.Json.Serialization;

namespace Rinha.Api;

public sealed class FraudRequest
{
    [JsonPropertyName("id")]
    public string? Id { get; set; }

    [JsonPropertyName("transaction")]
    public TransactionData Transaction { get; set; } = default!;

    [JsonPropertyName("customer")]
    public CustomerData Customer { get; set; } = default!;

    [JsonPropertyName("merchant")]
    public MerchantData Merchant { get; set; } = default!;

    [JsonPropertyName("terminal")]
    public TerminalData Terminal { get; set; } = default!;

    [JsonPropertyName("last_transaction")]
    public LastTransactionData? LastTransaction { get; set; }
}

public sealed class TransactionData
{
    [JsonPropertyName("amount")] public double Amount { get; set; }
    [JsonPropertyName("installments")] public int Installments { get; set; }
    [JsonPropertyName("requested_at")] public DateTimeOffset RequestedAt { get; set; }
}

public sealed class CustomerData
{
    [JsonPropertyName("avg_amount")] public double AvgAmount { get; set; }
    [JsonPropertyName("tx_count_24h")] public int TxCount24h { get; set; }
    [JsonPropertyName("known_merchants")] public List<string>? KnownMerchants { get; set; }
}

public sealed class MerchantData
{
    [JsonPropertyName("id")] public string Id { get; set; } = "";
    [JsonPropertyName("mcc")] public string Mcc { get; set; } = "";
    [JsonPropertyName("avg_amount")] public double AvgAmount { get; set; }
}

public sealed class TerminalData
{
    [JsonPropertyName("is_online")] public bool IsOnline { get; set; }
    [JsonPropertyName("card_present")] public bool CardPresent { get; set; }
    [JsonPropertyName("km_from_home")] public double KmFromHome { get; set; }
}

public sealed class LastTransactionData
{
    [JsonPropertyName("timestamp")] public DateTimeOffset Timestamp { get; set; }
    [JsonPropertyName("km_from_current")] public double KmFromCurrent { get; set; }
}

public sealed record FraudResponse(
    [property: JsonPropertyName("approved")] bool Approved,
    [property: JsonPropertyName("fraud_score")] float FraudScore);

[JsonSerializable(typeof(FraudRequest))]
[JsonSerializable(typeof(FraudResponse))]
[JsonSourceGenerationOptions(
    PropertyNamingPolicy = JsonKnownNamingPolicy.SnakeCaseLower,
    DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull)]
public partial class AppJsonContext : JsonSerializerContext
{
}
