using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;

namespace Rinha.Api.Scorers;

/// <summary>
/// Brute-force k-NN (k=5) using SIMD-accelerated squared euclidean distance.
/// Reference vectors are padded to 16 floats per row so each row is exactly
/// 2 × Vector256&lt;float&gt; loads with no tail handling.
/// </summary>
public sealed unsafe class BruteForceScorer : IFraudScorer
{
    private const int K = 5;
    private const int PaddedDimensions = 16;

    private readonly Dataset _dataset;

    public BruteForceScorer(Dataset dataset) => _dataset = dataset;

    public float Score(ReadOnlySpan<float> query)
    {
        if (query.Length < Dataset.Dimensions)
            throw new ArgumentException("Query vector too small", nameof(query));

        // Build the padded query vector (16 floats, last 2 = 0).
        Span<float> paddedQuery = stackalloc float[PaddedDimensions];
        for (int i = 0; i < Dataset.Dimensions; i++) paddedQuery[i] = query[i];
        paddedQuery[14] = 0f;
        paddedQuery[15] = 0f;

        Vector256<float> q0, q1;
        fixed (float* qPtr = paddedQuery)
        {
            q0 = Vector256.Load(qPtr);
            q1 = Vector256.Load(qPtr + 8);
        }

        Span<float> bestDist = stackalloc float[K];
        Span<int> bestIdx = stackalloc int[K];
        for (int i = 0; i < K; i++) { bestDist[i] = float.PositiveInfinity; bestIdx[i] = -1; }
        float worst = float.PositiveInfinity;

        var vectors = _dataset.VectorsPtr;
        int count = _dataset.Count;

        for (int i = 0; i < count; i++)
        {
            float* row = vectors + (long)i * PaddedDimensions;
            var r0 = Vector256.Load(row);
            var r1 = Vector256.Load(row + 8);
            var d0 = r0 - q0;
            var d1 = r1 - q1;
            var sum = (d0 * d0) + (d1 * d1);
            float dist = Vector256.Sum(sum);
            if (dist < worst)
            {
                InsertTopK(bestDist, bestIdx, dist, i);
                worst = bestDist[K - 1];
            }
        }

        var labels = _dataset.LabelsPtr;
        int frauds = 0;
        for (int i = 0; i < K; i++)
        {
            if (bestIdx[i] >= 0 && labels[bestIdx[i]] != 0) frauds++;
        }
        return frauds / (float)K;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void InsertTopK(Span<float> dist, Span<int> idx, float newDist, int newIdx)
    {
        int pos = K - 1;
        while (pos > 0 && dist[pos - 1] > newDist) pos--;
        for (int j = K - 1; j > pos; j--) { dist[j] = dist[j - 1]; idx[j] = idx[j - 1]; }
        dist[pos] = newDist;
        idx[pos] = newIdx;
    }
}
