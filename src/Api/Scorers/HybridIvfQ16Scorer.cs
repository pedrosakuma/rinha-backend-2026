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
public sealed class HybridIvfQ16Scorer : IFraudScorer
{
    private readonly IFraudScorer _ivf;
    private readonly IFraudScorer _brute;

    public HybridIvfQ16Scorer(IFraudScorer ivf, IFraudScorer brute)
    {
        _ivf = ivf;
        _brute = brute;
    }

    public float Score(ReadOnlySpan<float> query)
    {
        var s = _ivf.Score(query);
        // Borderline = the two discrete scores that are one neighbour-flip away from
        // the approval boundary. With K=5 each "vote" is worth ScoreStep (0.2), so the
        // values immediately on each side of ApprovalThreshold can be flipped by a
        // single misranked neighbour from the IVF approximation. Anything farther is
        // robust to single-vote IVF errors and we trust the cheap path.
        const float Step = PrecomputedFraudResponse.ScoreStep;
        const float T = PrecomputedFraudResponse.ApprovalThreshold;
        const float Eps = Step * 0.5f; // half-step tolerance for float compare
        if (MathF.Abs(s - (T - Step)) < Eps || MathF.Abs(s - T) < Eps)
            return _brute.Score(query);
        return s;
    }
}
