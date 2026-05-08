namespace Rinha.Api.Scorers;

/// <summary>
/// Hybrid IVF + Brute-Q16 scorer.
///
/// Strategy: run IVF first (fast, ~95% of the time correct). Approval threshold in
/// PrecomputedFraudResponse is <c>score &lt; 0.6f</c>, so the only IVF outputs that
/// can flip the approve/deny decision are the two adjacent to that boundary:
/// <c>0.4</c> (approved) and <c>0.6</c> (denied). When IVF returns one of those
/// borderline scores, rerun the query through the exact int16 brute-force scorer
/// (no IVF approximation, zero FN by construction) and use that result instead.
///
/// Cost: brute fires only on the borderline subset (~5% of queries empirically),
/// so amortised latency stays close to pure IVF while gaining brute's exactness on
/// every decision that actually matters.
/// </summary>
public sealed class HybridIvfQ16Scorer : IQ16FraudScorer
{
    private readonly IFraudScorer _ivf;
    private readonly IFraudScorer _brute;
    private readonly IQ16FraudScorer? _bruteQ16;

    public HybridIvfQ16Scorer(IFraudScorer ivf, IFraudScorer brute)
    {
        _ivf = ivf;
        _brute = brute;
        _bruteQ16 = brute as IQ16FraudScorer;
    }

    public float Score(ReadOnlySpan<float> query)
    {
        var s = _ivf.Score(query);
        if (IsBorderline(s))
            return _brute.Score(query);
        return s;
    }

    /// <summary>Q16 path: caller passes both representations so neither child has to
    /// re-quantize. IVF still consumes float; brute consumes the canonical short.</summary>
    public float ScoreQ16(ReadOnlySpan<short> query)
    {
        // Convert short → float once for IVF (bit-equal to Round4dp pipeline since
        // each lane is q / Q16Scale, the same value Round4dp would produce).
        Span<float> qf = stackalloc float[Dataset.Dimensions];
        for (int i = 0; i < Dataset.Dimensions; i++) qf[i] = query[i] * (1f / Dataset.Q16Scale);
        var s = _ivf.Score(qf);
        if (IsBorderline(s))
            return _bruteQ16 is { } b ? b.ScoreQ16(query) : _brute.Score(qf);
        return s;
    }

    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
    private static bool IsBorderline(float s)
    {
        // Borderline = the two discrete scores that are one neighbour-flip away from
        // the approval boundary. With K=5 each "vote" is worth ScoreStep (0.2), so the
        // values immediately on each side of ApprovalThreshold can be flipped by a
        // single misranked neighbour from the IVF approximation. Anything farther is
        // robust to single-vote IVF errors and we trust the cheap path.
        const float Step = PrecomputedFraudResponse.ScoreStep;
        const float T = PrecomputedFraudResponse.ApprovalThreshold;
        const float Eps = Step * 0.5f; // half-step tolerance for float compare
        return MathF.Abs(s - (T - Step)) < Eps || MathF.Abs(s - T) < Eps;
    }
}
