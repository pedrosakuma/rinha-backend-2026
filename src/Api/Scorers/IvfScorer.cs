using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Rinha.Api.Scorers;

/// <summary>
/// IVF (inverted-file) scorer with int8 quantized scan inside selected cells, then float recheck.
///
/// Pipeline:
///  1) Compute squared L2 from query to all NLIST centroids (small, fits in L1).
///  2) Pick top-NPROBE cells.
///  3) For each cell, scan its int8-quantized vectors with VPMADDWD pattern; keep top-K' candidates.
///  4) Recheck top-K' with exact float L2; keep top-K (5) → fraud ratio.
///
/// Layout assumption: data is reordered by cell at preprocess time; CellOffsetsPtr[c..c+1] gives
/// the contiguous range of vector indices belonging to cell c. Labels/Vectors/Q8 share index space.
///
/// Env: IVF_NPROBE (default 16), IVF_RERANK (default 32).
/// </summary>
public sealed unsafe class IvfScorer : IFraudScorer
{
    private const int K = 5;
    private const int PaddedDimensions = 16;

    private readonly Dataset _dataset;
    private readonly int _nProbe;
    private readonly int _kPrime;
    private readonly int _dimFilter;  // -1 = off; 0..13 = enable single-dim LB pruning
    private readonly int _dimFilter2; // optional 2nd dim added to LB

    public IvfScorer(Dataset dataset, int nProbe = 16, int kPrime = 32, int dimFilter = -1, int dimFilter2 = -1)
    {
        if (!dataset.HasIvf) throw new InvalidOperationException("Dataset has no IVF (centroids/offsets) view.");
        if (!dataset.HasQ8) throw new InvalidOperationException("Dataset has no Q8 view.");
        _dataset = dataset;
        _nProbe = Math.Clamp(nProbe, 1, dataset.NumCells);
        _kPrime = Math.Clamp(kPrime, K, 1024);
        _dimFilter = dimFilter is >= 0 and < Dataset.Dimensions ? dimFilter : -1;
        _dimFilter2 = dimFilter2 is >= 0 and < Dataset.Dimensions && dimFilter2 != _dimFilter ? dimFilter2 : -1;
    }

    public float Score(ReadOnlySpan<float> query)
    {
        if (query.Length < Dataset.Dimensions)
            throw new ArgumentException("Query vector too small", nameof(query));

        // 1) Quantize query to int8.
        Span<sbyte> qQ8 = stackalloc sbyte[PaddedDimensions];
        Span<float> paddedQuery = stackalloc float[PaddedDimensions];
        for (int i = 0; i < Dataset.Dimensions; i++)
        {
            paddedQuery[i] = query[i];
            int q = (int)MathF.Round(query[i] * Dataset.Q8Scale);
            if (q > 127) q = 127;
            else if (q < -128) q = -128;
            qQ8[i] = (sbyte)q;
        }

        int nlist = _dataset.NumCells;
        Span<float> centDist = stackalloc float[256]; // safe upper bound; nlist ≤ 256 by config
        if (nlist > 256)
        {
            // Defensive: heap fallback if user changes nlist beyond stack budget.
            centDist = new float[nlist];
        }

        // 2) Distances to all centroids.
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

        // 3) Pick top-NPROBE cells (small partial sort).
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

        // 4) Q8 scan inside selected cells, gather top-K'.
        Span<int> candDist = stackalloc int[_kPrime];
        Span<int> candIdx  = stackalloc int[_kPrime];
        for (int i = 0; i < _kPrime; i++) { candDist[i] = int.MaxValue; candIdx[i] = -1; }
        int q8Worst = int.MaxValue;

        var offsets = _dataset.CellOffsetsPtr;
        fixed (sbyte* qPtr = qQ8)
        {
            if (Avx2.IsSupported)
            {
                if (_dimFilter >= 0)
                    ScanCellsQ8Avx2DimFilter(qPtr, cells, offsets, candDist, candIdx, ref q8Worst, _dimFilter, _dimFilter2);
                else
                    ScanCellsQ8Avx2(qPtr, cells, offsets, candDist, candIdx, ref q8Worst);
            }
            else
                ScanCellsQ8Scalar(qPtr, cells, offsets, candDist, candIdx, ref q8Worst);
        }

        // 5) Float recheck of top-K' → top-K.
        Span<float> bestDist = stackalloc float[K];
        Span<int>   bestIdx  = stackalloc int[K];
        for (int i = 0; i < K; i++) { bestDist[i] = float.PositiveInfinity; bestIdx[i] = -1; }
        float worst = float.PositiveInfinity;

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

        // 6) Fraud ratio over exact top-K.
        var labels = _dataset.LabelsPtr;
        int frauds = 0;
        for (int i = 0; i < K; i++)
            if (bestIdx[i] >= 0 && labels[bestIdx[i]] != 0) frauds++;
        return frauds / (float)K;
    }

    private void ScanCellsQ8Avx2(
        sbyte* qPtr, ReadOnlySpan<int> cells, int* offsets,
        Span<int> candDist, Span<int> candIdx, ref int worstRef)
    {
        var qLow = Vector128.Load(qPtr);
        var qWide = Vector256.WidenLower(qLow.ToVector256());
        var sbase = _dataset.Q8VectorsPtr;
        int worst = worstRef;

        for (int ci = 0; ci < cells.Length; ci++)
        {
            int c = cells[ci];
            if (c < 0) break;
            int start = offsets[c];
            int end = offsets[c + 1];
            for (int i = start; i < end; i++)
            {
                sbyte* row = sbase + (long)i * PaddedDimensions;
                var r128 = Vector128.Load(row);
                var rWide = Vector256.WidenLower(r128.ToVector256());
                var diff = rWide - qWide;
                var prod = Avx2.MultiplyAddAdjacent(diff, diff);
                int dist = Vector256.Sum(prod);
                if (dist < worst)
                {
                    InsertTopKInt(candDist, candIdx, dist, i);
                    worst = candDist[candDist.Length - 1];
                }
            }
        }
        worstRef = worst;
    }

    /// <summary>
    /// Same as ScanCellsQ8Avx2 but with single-dim early-skip. For each row we first compute
    /// (r[D] - q[D])² as a strict lower bound on full Q8 dist; if it already exceeds the
    /// current top-K' worst, we skip the SIMD multiply entirely. Net win iff prune_rate &gt; ~50%.
    /// </summary>
    private void ScanCellsQ8Avx2DimFilter(
        sbyte* qPtr, ReadOnlySpan<int> cells, int* offsets,
        Span<int> candDist, Span<int> candIdx, ref int worstRef, int dimFilter, int dimFilter2)
    {
        var qLow = Vector128.Load(qPtr);
        var qWide = Vector256.WidenLower(qLow.ToVector256());
        var sbase = _dataset.Q8VectorsPtr;
        int worst = worstRef;
        int qd = qPtr[dimFilter];
        int qd2 = dimFilter2 >= 0 ? qPtr[dimFilter2] : 0;

        for (int ci = 0; ci < cells.Length; ci++)
        {
            int c = cells[ci];
            if (c < 0) break;
            int start = offsets[c];
            int end = offsets[c + 1];
            if (dimFilter2 >= 0)
            {
                for (int i = start; i < end; i++)
                {
                    sbyte* row = sbase + (long)i * PaddedDimensions;
                    int d1 = row[dimFilter] - qd;
                    int d2 = row[dimFilter2] - qd2;
                    int lb = d1 * d1 + d2 * d2;
                    if (lb >= worst) continue;

                    var r128 = Vector128.Load(row);
                    var rWide = Vector256.WidenLower(r128.ToVector256());
                    var diff = rWide - qWide;
                    var prod = Avx2.MultiplyAddAdjacent(diff, diff);
                    int dist = Vector256.Sum(prod);
                    if (dist < worst)
                    {
                        InsertTopKInt(candDist, candIdx, dist, i);
                        worst = candDist[candDist.Length - 1];
                    }
                }
            }
            else
            {
                for (int i = start; i < end; i++)
                {
                    sbyte* row = sbase + (long)i * PaddedDimensions;
                    int d1 = row[dimFilter] - qd;
                    int lb = d1 * d1;
                    if (lb >= worst) continue;

                    var r128 = Vector128.Load(row);
                    var rWide = Vector256.WidenLower(r128.ToVector256());
                    var diff = rWide - qWide;
                    var prod = Avx2.MultiplyAddAdjacent(diff, diff);
                    int dist = Vector256.Sum(prod);
                    if (dist < worst)
                    {
                        InsertTopKInt(candDist, candIdx, dist, i);
                        worst = candDist[candDist.Length - 1];
                    }
                }
            }
        }
        worstRef = worst;
    }

    private void ScanCellsQ8Scalar(
        sbyte* qPtr, ReadOnlySpan<int> cells, int* offsets,
        Span<int> candDist, Span<int> candIdx, ref int worstRef)
    {
        var sbase = _dataset.Q8VectorsPtr;
        int worst = worstRef;
        for (int ci = 0; ci < cells.Length; ci++)
        {
            int c = cells[ci];
            if (c < 0) break;
            int start = offsets[c];
            int end = offsets[c + 1];
            for (int i = start; i < end; i++)
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
