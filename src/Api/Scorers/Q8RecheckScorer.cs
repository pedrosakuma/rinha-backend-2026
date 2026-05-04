using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Rinha.Api.Scorers;

/// <summary>
/// Two-stage scorer: int8-quantized scan finds K' candidates, then float32 recheck
/// produces the exact top-K used for the fraud score.
///
/// Encoding: int8 = round(f * 127), clamp [-128, 127]. f∈[0,1] → [0,127];
/// sentinel f = -1 → -127. Distance is squared euclidean in 127-units; ranking is
/// monotonic with float L2 modulo quantization noise.
///
/// Per-row distance uses VPMADDWD (Avx2.MultiplyAddAdjacent): widen 16 sbyte → 16 int16,
/// subtract query, then madd(diff,diff) gives 8 int32 partials = sum of pairwise squares.
/// Pattern from FAISS/USearch L2 kernels.
///
/// K' is configurable via Q8_RERANK env var (default 32).
/// </summary>
public sealed unsafe class Q8RecheckScorer : IFraudScorer
{
    private const int K = 5;
    private const int PaddedDimensions = 16;

    private readonly Dataset _dataset;
    private readonly int _kPrime;

    public Q8RecheckScorer(Dataset dataset, int kPrime = 32)
    {
        if (!dataset.HasQ8)
            throw new InvalidOperationException("Dataset has no Q8 view; pass Q8_VECTORS_PATH at startup.");
        _dataset = dataset;
        _kPrime = Math.Clamp(kPrime, K, 256);
    }

    public float Score(ReadOnlySpan<float> query)
    {
        if (query.Length < Dataset.Dimensions)
            throw new ArgumentException("Query vector too small", nameof(query));

        // 1) Quantize query to int8 (16 bytes, padded).
        Span<sbyte> qQ8 = stackalloc sbyte[PaddedDimensions];
        for (int i = 0; i < Dataset.Dimensions; i++)
        {
            int q = (int)MathF.Round(query[i] * Dataset.Q8Scale);
            if (q > 127) q = 127;
            else if (q < -128) q = -128;
            qQ8[i] = (sbyte)q;
        }
        // qQ8[14], qQ8[15] already 0.

        // 2) First pass: int8 scan → top-K' candidates by quantized distance.
        Span<int>  candDist = stackalloc int[_kPrime];
        Span<int>  candIdx  = stackalloc int[_kPrime];
        for (int i = 0; i < _kPrime; i++) { candDist[i] = int.MaxValue; candIdx[i] = -1; }
        int q8Worst = int.MaxValue;

        fixed (sbyte* qPtr = qQ8)
        {
            if (Avx2.IsSupported)
                ScanQ8Avx2(qPtr, candDist, candIdx, ref q8Worst);
            else
                ScanQ8Scalar(qPtr, candDist, candIdx, ref q8Worst);
        }

        // 3) Recheck: compute exact float L2 for each candidate, keep top-K.
        Span<float> bestDist = stackalloc float[K];
        Span<int>   bestIdx  = stackalloc int[K];
        for (int i = 0; i < K; i++) { bestDist[i] = float.PositiveInfinity; bestIdx[i] = -1; }
        float worst = float.PositiveInfinity;

        Span<float> paddedQuery = stackalloc float[PaddedDimensions];
        for (int i = 0; i < Dataset.Dimensions; i++) paddedQuery[i] = query[i];

        fixed (float* qfPtr = paddedQuery)
        {
            var q0 = Vector256.Load(qfPtr);
            var q1 = Vector256.Load(qfPtr + 8);
            var vectors = _dataset.VectorsPtr;

            for (int c = 0; c < _kPrime; c++)
            {
                int idx = candIdx[c];
                if (idx < 0) break;
                float* row = vectors + (long)idx * PaddedDimensions;
                var r0 = Vector256.Load(row);
                var r1 = Vector256.Load(row + 8);
                var d0 = r0 - q0;
                var d1 = r1 - q1;
                var sum = (d0 * d0) + (d1 * d1);
                float dist = Vector256.Sum(sum);
                if (dist < worst)
                {
                    InsertTopK(bestDist, bestIdx, dist, idx);
                    worst = bestDist[K - 1];
                }
            }
        }

        // 4) Fraud ratio over the exact top-K.
        var labels = _dataset.LabelsPtr;
        int frauds = 0;
        for (int i = 0; i < K; i++)
            if (bestIdx[i] >= 0 && labels[bestIdx[i]] != 0) frauds++;
        return frauds / (float)K;
    }

    private void ScanQ8Avx2(sbyte* qPtr, Span<int> candDist, Span<int> candIdx, ref int worstRef)
    {
        // Widen the query (16 sbyte) once into a Vector256<short>.
        var qLow = Vector128.Load(qPtr);                         // 16 sbyte
        var qWide = Vector256.WidenLower(qLow.ToVector256());    // takes only the low 16 sbyte → 16 short
        // Note: we need the proper sign-extending widen of the low 128.
        // Vector256.WidenLower(Vector256<sbyte>) sign-extends the lower 16 sbyte to 16 short.
        // We'll re-do this more explicitly below.

        var sbase = _dataset.Q8VectorsPtr;
        int count = _dataset.Count;
        int worst = worstRef;

        for (int i = 0; i < count; i++)
        {
            sbyte* row = sbase + (long)i * PaddedDimensions;
            // Load 16 sbyte = 128 bits, sign-extend to Vector256<short>.
            var r128 = Vector128.Load(row);
            var rWide = Vector256.WidenLower(r128.ToVector256()); // 16 short
            var diff = rWide - qWide;
            // VPMADDWD: 16 short × 16 short → 8 int (sum of adjacent pairs of products).
            // Squared L2 = sum(diff_i^2) = madd(diff, diff) summed horizontally.
            var prod = Avx2.MultiplyAddAdjacent(diff, diff);     // Vector256<int>, 8 lanes
            int dist = Vector256.Sum(prod);
            if (dist < worst)
            {
                InsertTopKInt(candDist, candIdx, dist, i);
                worst = candDist[candDist.Length - 1];
            }
        }

        worstRef = worst;
    }

    private void ScanQ8Scalar(sbyte* qPtr, Span<int> candDist, Span<int> candIdx, ref int worstRef)
    {
        var sbase = _dataset.Q8VectorsPtr;
        int count = _dataset.Count;
        int worst = worstRef;

        for (int i = 0; i < count; i++)
        {
            sbyte* row = sbase + (long)i * PaddedDimensions;
            int dist = 0;
            for (int d = 0; d < Dataset.Dimensions; d++)
            {
                int diff = row[d] - qPtr[d];
                dist += diff * diff;
            }
            if (dist < worst)
            {
                InsertTopKInt(candDist, candIdx, dist, i);
                worst = candDist[candDist.Length - 1];
            }
        }

        worstRef = worst;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void InsertTopKInt(Span<int> dist, Span<int> idx, int newDist, int newIdx)
    {
        int n = dist.Length;
        int pos = n - 1;
        while (pos > 0 && dist[pos - 1] > newDist) pos--;
        for (int j = n - 1; j > pos; j--) { dist[j] = dist[j - 1]; idx[j] = idx[j - 1]; }
        dist[pos] = newDist;
        idx[pos] = newIdx;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void InsertTopK(Span<float> dist, Span<int> idx, float newDist, int newIdx)
    {
        int n = dist.Length;
        int pos = n - 1;
        while (pos > 0 && dist[pos - 1] > newDist) pos--;
        for (int j = n - 1; j > pos; j--) { dist[j] = dist[j - 1]; idx[j] = idx[j - 1]; }
        dist[pos] = newDist;
        idx[pos] = newIdx;
    }
}
