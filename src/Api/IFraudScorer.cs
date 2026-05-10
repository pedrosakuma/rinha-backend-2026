namespace Rinha.Api;

public interface IFraudScorer
{
    /// <summary>
    /// Returns the fraud score in [0,1] for a 14-dim padded query vector.
    /// </summary>
    float Score(ReadOnlySpan<float> query);

    /// <summary>
    /// Returns the integer fraud count in [0,K] (K=5) — bypasses the
    /// score = count/K float round-trip when callers will discretise back to int
    /// (e.g. <see cref="PrecomputedFraudResponse"/> selecting a pre-built response).
    /// Default implementation wraps <see cref="Score"/> and re-discretises.
    /// </summary>
    int ScoreCount(ReadOnlySpan<float> query)
    {
        float s = Score(query);
        int n = (int)MathF.Round(s * 5f);
        if (n < 0) n = 0;
        else if (n > 5) n = 5;
        return n;
    }
}

/// <summary>Optional capability: scorer accepts an int16-quantized query directly,
/// skipping the (short)Round(v * Q16Scale) re-quantization on the hot path. The
/// caller is expected to pass exactly <see cref="Dataset.Dimensions"/> shorts in
/// canonical units (Round(feature * Dataset.Q16Scale)).</summary>
public interface IQ16FraudScorer : IFraudScorer
{
    float ScoreQ16(ReadOnlySpan<short> query);
}

public static class ScorerFactory
{
    public static IFraudScorer Create(string name, Dataset dataset) => name.ToLowerInvariant() switch
    {
        "brute" or ""        => new Rinha.Api.Scorers.BruteForceScorer(dataset),
        "bruteq16" or "q16"  => new Rinha.Api.Scorers.BruteForceQ16Scorer(dataset),
        "q8"                 => new Rinha.Api.Scorers.Q8RecheckScorer(dataset, ParseInt(Environment.GetEnvironmentVariable("Q8_RERANK"), 32)),
        "ivf"                => new Rinha.Api.Scorers.IvfScorer(dataset,
                                    ParseInt(Environment.GetEnvironmentVariable("IVF_NPROBE"), 8),
                                    ParseInt(Environment.GetEnvironmentVariable("IVF_RERANK"), 32),
                                    earlyStop: Environment.GetEnvironmentVariable("IVF_EARLY_STOP") != "0",
                                    earlyStopPct: ParseInt(Environment.GetEnvironmentVariable("IVF_EARLY_STOP_PCT"), 75),
                                    bboxRepair: Environment.GetEnvironmentVariable("IVF_BBOX_REPAIR") == "1",
                                    earlyStopPctEarly: ParseInt(Environment.GetEnvironmentVariable("IVF_EARLY_STOP_PCT_EARLY"), 0),
                                    bboxGuided: Environment.GetEnvironmentVariable("IVF_BBOX_GUIDED") == "1"),
        "ivf-blocked" or "ivfblocked"
                             => new Rinha.Api.Scorers.IvfBlockedScorer(dataset,
                                    ParseInt(Environment.GetEnvironmentVariable("IVF_BLOCKED_NPROBE"), 4)),
        _ => throw new ArgumentException($"Unknown scorer '{name}'. Known: brute, bruteq16, q8, ivf, ivf-blocked")
    };

    private static int ParseInt(string? s, int defaultValue)
        => int.TryParse(s, out var v) ? v : defaultValue;
}
