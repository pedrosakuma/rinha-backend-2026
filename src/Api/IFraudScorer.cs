namespace Rinha.Api;

public interface IFraudScorer
{
    /// <summary>
    /// Returns the fraud score in [0,1] for a 14-dim padded query vector.
    /// </summary>
    float Score(ReadOnlySpan<float> query);
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
        "fma"                => new Rinha.Api.Scorers.FmaBruteForceScorer(dataset),
        "q8"                 => new Rinha.Api.Scorers.Q8RecheckScorer(dataset, ParseInt(Environment.GetEnvironmentVariable("Q8_RERANK"), 32)),
        "ivf"                => new Rinha.Api.Scorers.IvfScorer(dataset,
                                    ParseInt(Environment.GetEnvironmentVariable("IVF_NPROBE"), 16),
                                    ParseInt(Environment.GetEnvironmentVariable("IVF_RERANK"), 32),
                                    ParseInt(Environment.GetEnvironmentVariable("IVF_DIM_FILTER"), -1),
                                    ParseInt(Environment.GetEnvironmentVariable("IVF_DIM_FILTER2"), -1),
                                    Environment.GetEnvironmentVariable("IVF_EARLY_STOP") == "1",
                                    ParseInt(Environment.GetEnvironmentVariable("IVF_EARLY_STOP_PCT"), 75),
                                    Environment.GetEnvironmentVariable("IVF_BBOX_REPAIR") == "1",
                                    ParseInt(Environment.GetEnvironmentVariable("IVF_EARLY_STOP_PCT_EARLY"), 0),
                                    ParseInt(Environment.GetEnvironmentVariable("IVF_SCALAR_ABORT"), 0),
                                    Environment.GetEnvironmentVariable("IVF_DENSITY_ORDER") == "1"),
        "ivfpq"              => new Rinha.Api.Scorers.IvfPqScorer(dataset,
                                    ParseInt(Environment.GetEnvironmentVariable("IVF_NPROBE"), 96),
                                    ParseInt(Environment.GetEnvironmentVariable("IVF_RERANK"), 64)),
        "hybrid"             => new Rinha.Api.Scorers.HybridIvfQ16Scorer(
                                    Create("ivf", dataset),
                                    Create("bruteq16", dataset)),
        _ => throw new ArgumentException($"Unknown scorer '{name}'. Known: brute, bruteq16, fma, q8, ivf, ivfpq, hybrid")
    };

    private static int ParseInt(string? s, int defaultValue)
        => int.TryParse(s, out var v) ? v : defaultValue;
}
