using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Rinha.Api.Scorers;

/// <summary>
/// Brute-force k-NN (k=5) over the int16-quantized reference vectors (Dataset.Q16VectorsPtr).
///
/// Layout: each row is exactly 16 shorts (14 dims + 2 zero pad) = 32 bytes — one Vector256&lt;short&gt;
/// load per row, half the bandwidth of float32 brute. Distance computed via VPMADDWD
/// (MultiplyAddAdjacent): squares + pairwise-sum int16 lanes into 8 int32 lanes in one op.
/// Final reduction widens to int64 to avoid overflow when scanning very far rows
/// (max diff² × 14 ~ 5.6e9, exceeds int32).
///
/// Bounded skip: keep current worst (in int64²-of-quantized-units) and skip InsertTopK when
/// dist >= worst. With cpuset isolating the core, the 1.7MB Q16 working set fits in L2/L3
/// without contention, making this competitive with IVF while avoiding any FN.
/// </summary>
public sealed unsafe class BruteForceQ16Scorer : IQ16FraudScorer
{
    private const int K = 5;
    private const int PaddedDimensions = 16;

    private readonly Dataset _dataset;

    public BruteForceQ16Scorer(Dataset dataset)
    {
        if (!dataset.HasQ16)
            throw new InvalidOperationException("BruteForceQ16Scorer requires Q16 layout (VECTORS_Q16_PATH)");
        _dataset = dataset;
    }

    public float Score(ReadOnlySpan<float> query)
    {
        if (query.Length < Dataset.Dimensions)
            throw new ArgumentException("Query vector too small", nameof(query));

        // Quantize query to 16 shorts (lanes 14,15 = 0 pad).
        Span<short> qq = stackalloc short[PaddedDimensions];
        for (int i = 0; i < Dataset.Dimensions; i++)
        {
            int q = (int)MathF.Round(query[i] * Dataset.Q16Scale);
            if (q > 32767) q = 32767;
            else if (q < -32768) q = -32768;
            qq[i] = (short)q;
        }
        qq[14] = 0;
        qq[15] = 0;
        return ScoreQ16Internal(qq);
    }

    /// <summary>Q16 fast path: caller passes the 14 canonical shorts directly. The
    /// scorer fills the 2 trailing pad lanes itself (the row layout in
    /// <see cref="Dataset.Q16VectorsPtr"/> requires exactly 16 lanes per row, with
    /// lanes 14,15 = 0 in references — so the query also pads with 0 to match).</summary>
    public float ScoreQ16(ReadOnlySpan<short> query)
    {
        if (query.Length < Dataset.Dimensions)
            throw new ArgumentException("Query vector too small", nameof(query));

        Span<short> qq = stackalloc short[PaddedDimensions];
        // Copy 14 canonical lanes; lanes 14,15 stay 0 (Span<short> stackalloc is zero-init).
        query.Slice(0, Dataset.Dimensions).CopyTo(qq);
        return ScoreQ16Internal(qq);
    }

    private float ScoreQ16Internal(Span<short> qq)
    {
        Span<long> bestDist = stackalloc long[K];
        Span<int> bestIdx = stackalloc int[K];
        for (int i = 0; i < K; i++) { bestDist[i] = long.MaxValue; bestIdx[i] = -1; }

        fixed (short* qPtr = qq)
        fixed (long* bdPtr = bestDist)
        fixed (int* biPtr = bestIdx)
        {
            if (Avx2.IsSupported && _dataset.HasQ16Soa)
                ScanAvx2Soa(qPtr, bdPtr, biPtr);
            else if (Avx2.IsSupported)
                ScanAvx2(qPtr, bdPtr, biPtr);
            else
                ScanScalar(qPtr, bdPtr, biPtr);

            var labels = _dataset.LabelsPtr;
            int frauds = 0;
            for (int i = 0; i < K; i++)
            {
                int idx = biPtr[i];
                if (idx >= 0 && labels[idx] != 0) frauds++;
            }
            return frauds / (float)K;
        }
    }

    private void ScanAvx2Soa(short* qPtr, long* bestDist, int* bestIdx)
    {
        long worst = long.MaxValue;
        short* soaPtr = _dataset.Q16SoaPtr;
        int count = _dataset.Count;
        int fullBatches = count & ~15; // round down to multiple of 16

        Span<int> dists = stackalloc int[16];
        fixed (int* distsPtr = dists)
        {
            int batchStart = 0;
            for (; batchStart < fullBatches; batchStart += 16)
            {
                var acc0 = Vector256<int>.Zero; // distances for candidates 0-7
                var acc1 = Vector256<int>.Zero; // distances for candidates 8-15

                for (int d = 0; d < Dataset.Dimensions; d++)
                {
                    // Broadcast single query dim to all 16 lanes (vpbroadcastw).
                    var qd = Vector256.Create(qPtr[d]);
                    // Load 16 reference shorts from the column for dim d (vpsubw target).
                    short* col = soaPtr + (long)d * count + batchStart;
                    var refs = Vector256.Load(col);
                    // vpsubw: diff[k] = ref[k] - query[d]
                    var diff = Avx2.Subtract(refs, qd);
                    // Sign-extend 16 shorts → two Vector256<int> of 8 int32s each (vpmovsxwd).
                    var (lo, hi) = Vector256.Widen(diff);
                    // vpmulld + vpaddd: square and accumulate in int32.
                    // Max diff=10000 → diff²=1e8 → 14×1e8=1.4e9 < int32.MaxValue ✓
                    acc0 = Avx2.Add(acc0, Avx2.MultiplyLow(lo, lo));
                    acc1 = Avx2.Add(acc1, Avx2.MultiplyLow(hi, hi));
                }

                Vector256.Store(acc0, distsPtr);
                Vector256.Store(acc1, distsPtr + 8);

                for (int k = 0; k < 16; k++)
                {
                    long dist = distsPtr[k];
                    if (dist < worst)
                        worst = InsertTopK(bestDist, bestIdx, dist, batchStart + k);
                }
            }
            // Scalar tail for remaining count % 16 elements.
            for (int i = batchStart; i < count; i++)
            {
                long dist = 0;
                for (int d = 0; d < Dataset.Dimensions; d++)
                {
                    int diff = soaPtr[(long)d * count + i] - qPtr[d];
                    dist += (long)diff * diff;
                }
                if (dist < worst)
                    worst = InsertTopK(bestDist, bestIdx, dist, i);
            }
        }
    }

    private void ScanAvx2(short* qPtr, long* bestDist, int* bestIdx)
    {
        long worst = long.MaxValue;
        var vectors = _dataset.Q16VectorsPtr;
        int count = _dataset.Count;

        var qv = Vector256.Load(qPtr); // 16 shorts

        for (int i = 0; i < count; i++)
        {
            short* row = vectors + (long)i * PaddedDimensions;
            var rv = Vector256.Load(row);
            var diff = Avx2.Subtract(rv, qv);
            // VPMADDWD: per pair (a*b + c*d) widening int16→int32 → 8 int32 lanes.
            // Each lane <= 2 * 32767² ≈ 2.1e9, no overflow.
            var sq = Avx2.MultiplyAddAdjacent(diff, diff);
            // Sum of 8 int32s could reach ~1.7e10 → widen to int64 before horizontal sum.
            long dist = HorizontalSumInt64(sq);
            if (dist < worst)
            {
                worst = InsertTopK(bestDist, bestIdx, dist, i);
            }
        }
    }

    private void ScanScalar(short* qPtr, long* bestDist, int* bestIdx)
    {
        long worst = long.MaxValue;
        var vectors = _dataset.Q16VectorsPtr;
        int count = _dataset.Count;

        for (int i = 0; i < count; i++)
        {
            short* row = vectors + (long)i * PaddedDimensions;
            long dist = 0;
            for (int d = 0; d < PaddedDimensions; d++)
            {
                int diff = row[d] - qPtr[d];
                dist += (long)diff * diff;
            }
            if (dist < worst)
            {
                worst = InsertTopK(bestDist, bestIdx, dist, i);
            }
        }
    }

    /// <summary>Sum 8 int32 lanes safely into int64 (avoids overflow when summing
    /// large-magnitude squared differences across the full reference set).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static long HorizontalSumInt64(Vector256<int> v)
    {
        // Widen 8 int32 → 2 × Vector256<long> (4 longs each) and add.
        var (lo, hi) = Vector256.Widen(v);
        var s = lo + hi; // 4 int64 lanes
        var sLo = s.GetLower();
        var sHi = s.GetUpper();
        var pair = Sse2.Add(sLo, sHi); // 2 int64 lanes
        return pair.ToScalar() + pair.GetElement(1);
    }

    /// <summary>Insert (newDist, newIdx) into the descending Top-K and return the new worst (= dist[K-1]).
    /// Uses raw pointers — no bounds checks, no Span overhead. Caller guarantees buffers have K elements.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static long InsertTopK(long* dist, int* idx, long newDist, int newIdx)
    {
        int pos = K - 1;
        while (pos > 0 && dist[pos - 1] > newDist) pos--;
        for (int j = K - 1; j > pos; j--) { dist[j] = dist[j - 1]; idx[j] = idx[j - 1]; }
        dist[pos] = newDist;
        idx[pos] = newIdx;
        return dist[K - 1];
    }
}
