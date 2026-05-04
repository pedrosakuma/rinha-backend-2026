namespace Rinha.Api;

public interface IFraudScorer
{
    /// <summary>
    /// Returns the fraud score in [0,1] for a 14-dim padded query vector.
    /// </summary>
    float Score(ReadOnlySpan<float> query);
}

public static class ScorerFactory
{
    public static IFraudScorer Create(string name, Dataset dataset) => name.ToLowerInvariant() switch
    {
        "brute" or ""        => new Rinha.Api.Scorers.BruteForceScorer(dataset),
        "fma"                => new Rinha.Api.Scorers.FmaBruteForceScorer(dataset),
        "q8"                 => new Rinha.Api.Scorers.Q8RecheckScorer(dataset, ParseInt(Environment.GetEnvironmentVariable("Q8_RERANK"), 32)),
        _ => throw new ArgumentException($"Unknown scorer '{name}'. Known: brute, fma, q8")
    };

    private static int ParseInt(string? s, int defaultValue)
        => int.TryParse(s, out var v) ? v : defaultValue;
}
