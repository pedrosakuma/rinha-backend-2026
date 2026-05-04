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
        _ => throw new ArgumentException($"Unknown scorer '{name}'. Known: brute")
    };
}
