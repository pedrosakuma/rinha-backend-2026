using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;

namespace Rinha.Api.Scorers;

/// <summary>
/// IVF + PQ (asymmetric distance computation) scorer with float recheck.
///
/// Pipeline:
///  1) Compute squared L2 from query to all NLIST IVF centroids.
///  2) Pick top-NPROBE cells.
///  3) Build per-query LUT[M][ksub] of dist² between each sub-query and each PQ centroid.
///  4) For each row in selected cells, sum M LUT lookups → approximate dist; keep top-K' by ADC.
///  5) Recheck top-K' with exact float L2 → top-K (5) → fraud ratio.
///
/// Requires Dataset.HasIvf and Dataset.HasPq.
///
/// Env: IVF_NPROBE (default 96), IVF_RERANK (default 64).
/// </summary>
public sealed unsafe class IvfPqScorer : IFraudScorer
{
    private const int K = 5;
    private const int Dimensions = Dataset.Dimensions;
    private const int PaddedDimensions = Dataset.PaddedDimensions;

    private readonly Dataset _dataset;
    private readonly int _nProbe;
    private readonly int _kPrime;
    private readonly int _M;
    private readonly int _ksub;
    private readonly int _dsub;

    public IvfPqScorer(Dataset dataset, int nProbe = 96, int kPrime = 64)
    {
        if (!dataset.HasIvf) throw new InvalidOperationException("Dataset has no IVF view.");
        if (!dataset.HasPq) throw new InvalidOperationException("Dataset has no PQ view.");
        _dataset = dataset;
        _nProbe = Math.Clamp(nProbe, 1, dataset.NumCells);
        _kPrime = Math.Clamp(kPrime, K, 1024);
        _M = dataset.PqM;
        _ksub = dataset.PqKsub;
        _dsub = Dimensions / _M;
        if (Dimensions % _M != 0)
            throw new InvalidOperationException($"PQ M={_M} must divide D={Dimensions}");
    }

    public float Score(ReadOnlySpan<float> query)
    {
        if (query.Length < Dimensions)
            throw new ArgumentException("Query vector too small", nameof(query));

        Span<float> paddedQuery = stackalloc float[PaddedDimensions];
        for (int i = 0; i < Dimensions; i++) paddedQuery[i] = query[i];

        int nlist = _dataset.NumCells;
        Span<float> centDist = nlist <= 256 ? stackalloc float[256] : new float[nlist];

        // 1) Distances to all IVF centroids.
        fixed (float* qfPtr = paddedQuery)
        {
            var q0 = Vector256.Load(qfPtr);
            var q1 = Vector256.Load(qfPtr + 8);
            var centBase = _dataset.CentroidsPtr;
            for (int c = 0; c < nlist; c++)
            {
                float* cp = centBase + (long)c * PaddedDimensions;
                var k0 = Vector256.Load(cp);
                var k1 = Vector256.Load(cp + 8);
                var d0 = k0 - q0;
                var d1 = k1 - q1;
                var s = (d0 * d0) + (d1 * d1);
                centDist[c] = Vector256.Sum(s);
            }
        }

        // 2) Top-NPROBE cells (insert-sort).
        Span<int>   cells     = stackalloc int[_nProbe];
        Span<float> cellsDist = stackalloc float[_nProbe];
        for (int i = 0; i < _nProbe; i++) { cells[i] = -1; cellsDist[i] = float.PositiveInfinity; }
        float cellsWorst = float.PositiveInfinity;
        for (int c = 0; c < nlist; c++)
        {
            float d = centDist[c];
            if (d < cellsWorst)
            {
                int pos = _nProbe - 1;
                while (pos > 0 && cellsDist[pos - 1] > d) pos--;
                for (int j = _nProbe - 1; j > pos; j--) { cellsDist[j] = cellsDist[j - 1]; cells[j] = cells[j - 1]; }
                cellsDist[pos] = d;
                cells[pos] = c;
                cellsWorst = cellsDist[_nProbe - 1];
            }
        }

        // 3) Build LUT[M][ksub] = dist²(query_subvec[m], codebook[m][k]).
        //    Default 7 × 256 × 4B = 7 KB → fits in L1.
        Span<float> lut = stackalloc float[_M * _ksub];
        var codebooks = _dataset.PqCodebooksPtr;
        for (int m = 0; m < _M; m++)
        {
            float* cb = codebooks + (long)m * _ksub * _dsub;
            float* lutM = (float*)Unsafe.AsPointer(ref lut[m * _ksub]);
            int subOff = m * _dsub;
            // dsub default = 2, unroll.
            float q0 = paddedQuery[subOff];
            float q1 = _dsub > 1 ? paddedQuery[subOff + 1] : 0;
            if (_dsub == 2)
            {
                for (int k = 0; k < _ksub; k++)
                {
                    float dx = cb[k * 2] - q0;
                    float dy = cb[k * 2 + 1] - q1;
                    lutM[k] = dx * dx + dy * dy;
                }
            }
            else
            {
                for (int k = 0; k < _ksub; k++)
                {
                    float d2 = 0;
                    for (int d = 0; d < _dsub; d++)
                    {
                        float diff = cb[k * _dsub + d] - paddedQuery[subOff + d];
                        d2 += diff * diff;
                    }
                    lutM[k] = d2;
                }
            }
        }

        // 4) ADC scan inside selected cells, gather top-K' by approximate dist.
        Span<float> candDist = stackalloc float[_kPrime];
        Span<int>   candIdx  = stackalloc int[_kPrime];
        for (int i = 0; i < _kPrime; i++) { candDist[i] = float.PositiveInfinity; candIdx[i] = -1; }
        float adcWorst = float.PositiveInfinity;

        var offsets = _dataset.CellOffsetsPtr;
        var pqCodes = _dataset.PqCodesPtr;
        int M = _M;
        fixed (float* lutPtr = lut)
        {
            for (int ci = 0; ci < cells.Length; ci++)
            {
                int c = cells[ci];
                if (c < 0) break;
                int start = offsets[c];
                int end = offsets[c + 1];
                if (M == 7)
                {
                    // Unrolled hot path.
                    for (int i = start; i < end; i++)
                    {
                        byte* row = pqCodes + (long)i * 7;
                        float dist =
                              lutPtr[0 * 256 + row[0]]
                            + lutPtr[1 * 256 + row[1]]
                            + lutPtr[2 * 256 + row[2]]
                            + lutPtr[3 * 256 + row[3]]
                            + lutPtr[4 * 256 + row[4]]
                            + lutPtr[5 * 256 + row[5]]
                            + lutPtr[6 * 256 + row[6]];
                        if (dist < adcWorst)
                        {
                            InsertTopKFloat(candDist, candIdx, dist, i);
                            adcWorst = candDist[_kPrime - 1];
                        }
                    }
                }
                else
                {
                    int ksub = _ksub;
                    for (int i = start; i < end; i++)
                    {
                        byte* row = pqCodes + (long)i * M;
                        float dist = 0;
                        for (int m = 0; m < M; m++)
                            dist += lutPtr[m * ksub + row[m]];
                        if (dist < adcWorst)
                        {
                            InsertTopKFloat(candDist, candIdx, dist, i);
                            adcWorst = candDist[_kPrime - 1];
                        }
                    }
                }
            }
        }

        // 5) Float recheck of top-K' → top-K (5).
        Span<float> bestDist = stackalloc float[K];
        Span<int>   bestIdx  = stackalloc int[K];
        for (int i = 0; i < K; i++) { bestDist[i] = float.PositiveInfinity; bestIdx[i] = -1; }
        float worst = float.PositiveInfinity;

        fixed (float* qfPtr = paddedQuery)
        {
            var q0v = Vector256.Load(qfPtr);
            var q1v = Vector256.Load(qfPtr + 8);
            var vectors = _dataset.VectorsPtr;
            for (int c = 0; c < _kPrime; c++)
            {
                int idx = candIdx[c];
                if (idx < 0) break;
                float* row = vectors + (long)idx * PaddedDimensions;
                var r0 = Vector256.Load(row);
                var r1 = Vector256.Load(row + 8);
                var d0 = r0 - q0v;
                var d1 = r1 - q1v;
                var sum = (d0 * d0) + (d1 * d1);
                float dist = Vector256.Sum(sum);
                if (dist < worst)
                {
                    InsertTopKFloat(bestDist, bestIdx, dist, idx);
                    worst = bestDist[K - 1];
                }
            }
        }

        var labels = _dataset.LabelsPtr;
        int frauds = 0;
        for (int i = 0; i < K; i++)
            if (bestIdx[i] >= 0 && labels[bestIdx[i]] != 0) frauds++;
        return frauds / (float)K;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void InsertTopKFloat(Span<float> dist, Span<int> idx, float newDist, int newIdx)
    {
        int n = dist.Length;
        int pos = n - 1;
        while (pos > 0 && dist[pos - 1] > newDist) pos--;
        for (int j = n - 1; j > pos; j--) { dist[j] = dist[j - 1]; idx[j] = idx[j - 1]; }
        dist[pos] = newDist;
        idx[pos] = newIdx;
    }
}
