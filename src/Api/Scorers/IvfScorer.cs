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
    private readonly bool _earlyStop; // class-aware early-stop (J5)
    private readonly int _earlyStopPct; // checkpoint % of nProbe (default 75)
    private readonly int _earlyStopPctEarly; // J9: optional 2nd, earlier checkpoint (e.g. 25). 0 = disabled.
    private readonly bool _bboxRepair; // J6: bbox LB exact repair
    private readonly bool _scalarAbort; // J10: scalar early-abort dim-by-dim per row

    public IvfScorer(Dataset dataset, int nProbe = 16, int kPrime = 32, int dimFilter = -1, int dimFilter2 = -1, bool earlyStop = false, int earlyStopPct = 75, bool bboxRepair = false, int earlyStopPctEarly = 0, bool scalarAbort = false)
    {
        if (!dataset.HasIvf) throw new InvalidOperationException("Dataset has no IVF (centroids/offsets) view.");
        if (!dataset.HasQ8) throw new InvalidOperationException("Dataset has no Q8 view.");
        _dataset = dataset;
        _nProbe = Math.Clamp(nProbe, 1, dataset.NumCells);
        _kPrime = Math.Clamp(kPrime, K, 1024);
        _dimFilter = dimFilter is >= 0 and < Dataset.Dimensions ? dimFilter : -1;
        _dimFilter2 = dimFilter2 is >= 0 and < Dataset.Dimensions && dimFilter2 != _dimFilter ? dimFilter2 : -1;
        _earlyStop = earlyStop;
        _earlyStopPct = Math.Clamp(earlyStopPct, 10, 95);
        _earlyStopPctEarly = earlyStopPctEarly > 0 && earlyStopPctEarly < _earlyStopPct ? earlyStopPctEarly : 0;
        _bboxRepair = bboxRepair && dataset.HasIvfBbox;
        _scalarAbort = scalarAbort;
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

        Span<float> bestDist = stackalloc float[K];
        Span<int>   bestIdx  = stackalloc int[K];
        for (int i = 0; i < K; i++) { bestDist[i] = float.PositiveInfinity; bestIdx[i] = -1; }
        bool earlyStopped = false;

        var offsets = _dataset.CellOffsetsPtr;
        var labels = _dataset.LabelsPtr;
        fixed (sbyte* qPtr = qQ8)
        {
            if (Avx2.IsSupported)
            {
                if (_dimFilter >= 0)
                {
                    ScanCellsQ8Avx2DimFilter(qPtr, cells, offsets, candDist, candIdx, ref q8Worst, _dimFilter, _dimFilter2);
                }
                else if (_earlyStop)
                {
                    // Class-aware early-stop: process cells up to a checkpoint; if top-K
                    // is class-unanimous AND worst float dist < next-cell centroid dist,
                    // skip remaining cells (binary "approved" outcome locked).
                    // Checkpoint at 75%: smaller savings on easy queries, but cheaper
                    // overhead on the ~3% ambiguous tail (which dominates p99 under
                    // saturation).
                    int checkpoint = (_nProbe * _earlyStopPct) / 100;
                    int earlyCheckpoint = (_nProbe * _earlyStopPctEarly) / 100;
                    int curStart = 0;

                    // J9: optional very-early checkpoint (e.g. 25%) with the same margin rule.
                    // Saves ~50-70% of Q8 scan on "easy" queries (clear class, big margin).
                    if (earlyCheckpoint >= 1 && earlyCheckpoint < checkpoint)
                    {
                        ScanCellsQ8Avx2Range(qPtr, cells, curStart, earlyCheckpoint, offsets, candDist, candIdx, ref q8Worst);
                        RerankFloat(paddedQuery, candDist, candIdx, bestDist, bestIdx);
                        int frauds0 = 0, valid0 = 0;
                        for (int i = 0; i < K; i++)
                        {
                            if (bestIdx[i] < 0) continue;
                            valid0++;
                            if (labels[bestIdx[i]] != 0) frauds0++;
                        }
                        bool unanimous0 = (valid0 == K) && (frauds0 == 0 || frauds0 == K);
                        bool marginOk0 = unanimous0 && bestDist[K - 1] < cellsDist[earlyCheckpoint];
                        if (marginOk0)
                        {
                            earlyStopped = true;
                        }
                        else
                        {
                            curStart = earlyCheckpoint;
                        }
                    }

                    if (!earlyStopped && checkpoint >= 1 && _nProbe > checkpoint)
                    {
                        ScanCellsQ8Avx2Range(qPtr, cells, curStart, checkpoint, offsets, candDist, candIdx, ref q8Worst);

                        // Partial rerank using current candDist/candIdx → bestDist/bestIdx.
                        // Reset bestIdx so we re-pick from current candidate pool.
                        for (int i = 0; i < K; i++) { bestDist[i] = float.PositiveInfinity; bestIdx[i] = -1; }
                        RerankFloat(paddedQuery, candDist, candIdx, bestDist, bestIdx);
                        int frauds = 0, valid = 0;
                        for (int i = 0; i < K; i++)
                        {
                            if (bestIdx[i] < 0) continue;
                            valid++;
                            if (labels[bestIdx[i]] != 0) frauds++;
                        }
                        bool unanimous = (valid == K) && (frauds == 0 || frauds == K);
                        bool marginOk = unanimous && bestDist[K - 1] < cellsDist[checkpoint];
                        if (marginOk)
                        {
                            earlyStopped = true;
                        }
                        else
                        {
                            ScanCellsQ8Avx2Range(qPtr, cells, checkpoint, _nProbe, offsets, candDist, candIdx, ref q8Worst);
                        }
                    }
                    else if (!earlyStopped)
                    {
                        ScanCellsQ8Avx2(qPtr, cells, offsets, candDist, candIdx, ref q8Worst);
                    }
                }
                else
                {
                    ScanCellsQ8Avx2(qPtr, cells, offsets, candDist, candIdx, ref q8Worst);
                }
            }
            else
                ScanCellsQ8Scalar(qPtr, cells, offsets, candDist, candIdx, ref q8Worst);
        }

        // 5) Float recheck of top-K' → top-K (skipped if already done by early-stop).
        if (!earlyStopped)
        {
            for (int i = 0; i < K; i++) { bestDist[i] = float.PositiveInfinity; bestIdx[i] = -1; }
            RerankFloat(paddedQuery, candDist, candIdx, bestDist, bestIdx);
        }

        // 5.5) J6: bbox lower-bound exact repair. For each non-seed cluster, compute float
        // LB = Σ max(0, max(min[c,j]-q[j], q[j]-max[c,j]))². If LB ≤ worst float dist, the
        // cluster MAY contain a vector beating the current top-K, so we Q8-scan it; else skip.
        // Preserves exact recall (LB is admissible). Skipped when class-aware early-stop fired
        // (decision already locked by margin proof).
        if (_bboxRepair && !earlyStopped)
        {
            int nlistLocal = _dataset.NumCells;
            Span<bool> isSeed = stackalloc bool[nlistLocal];
            for (int i = 0; i < cells.Length; i++)
            {
                int c = cells[i];
                if (c >= 0) isSeed[c] = true;
            }

            Span<int> repairCells = stackalloc int[nlistLocal];
            int repairCount = 0;
            float worstD = bestDist[K - 1];
            fixed (float* qfPtr = paddedQuery)
            {
                for (int c = 0; c < nlistLocal; c++)
                {
                    if (isSeed[c]) continue;
                    if (BboxLowerBoundSquared(qfPtr, c) <= worstD)
                    {
                        repairCells[repairCount++] = c;
                    }
                }
            }

            if (repairCount > 0)
            {
                fixed (sbyte* qPtr = qQ8)
                {
                    ScanCellsQ8Avx2Range(qPtr, repairCells, 0, repairCount, offsets, candDist, candIdx, ref q8Worst);
                }
                for (int i = 0; i < K; i++) { bestDist[i] = float.PositiveInfinity; bestIdx[i] = -1; }
                RerankFloat(paddedQuery, candDist, candIdx, bestDist, bestIdx);
            }
        }

        // 6) Fraud ratio over exact top-K.
        int totalFrauds = 0;
        for (int i = 0; i < K; i++)
            if (bestIdx[i] >= 0 && labels[bestIdx[i]] != 0) totalFrauds++;
        return totalFrauds / (float)K;
    }

    /// <summary>
    /// Float lower bound on squared L2 distance from query to any point in cluster <paramref name="c"/>,
    /// using the per-cell bounding box (min/max per dim). Admissible: returns ≤ true min distance,
    /// so safe for prune-only decisions ("if LB > worst_d → cluster cannot improve top-K").
    /// AVX2: two 256-bit loads × (max(0, max(min-q, q-max)))² per 8 dims.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private float BboxLowerBoundSquared(float* qPtr, int c)
    {
        float* mn = _dataset.IvfBboxMinPtr + (long)c * PaddedDimensions;
        float* mx = _dataset.IvfBboxMaxPtr + (long)c * PaddedDimensions;
        var q0 = Vector256.Load(qPtr);
        var q1 = Vector256.Load(qPtr + 8);
        var mn0 = Vector256.Load(mn);
        var mn1 = Vector256.Load(mn + 8);
        var mx0 = Vector256.Load(mx);
        var mx1 = Vector256.Load(mx + 8);
        var d0 = Vector256.Max(Vector256.Max(mn0 - q0, q0 - mx0), Vector256<float>.Zero);
        var d1 = Vector256.Max(Vector256.Max(mn1 - q1, q1 - mx1), Vector256<float>.Zero);
        var s = (d0 * d0) + (d1 * d1);
        return Vector256.Sum(s);
    }

    private void RerankFloat(
        ReadOnlySpan<float> paddedQuery,
        ReadOnlySpan<int> candDist, ReadOnlySpan<int> candIdx,
        Span<float> bestDist, Span<int> bestIdx)
    {
        float worst = bestDist[K - 1];
        fixed (float* qfPtr = paddedQuery)
        {
            var q0 = Vector256.Load(qfPtr);
            var q1 = Vector256.Load(qfPtr + 8);
            var vectors = _dataset.VectorsPtr;
            for (int c = 0; c < candIdx.Length; c++)
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
    }

    private void ScanCellsQ8Avx2(
        sbyte* qPtr, ReadOnlySpan<int> cells, int* offsets,
        Span<int> candDist, Span<int> candIdx, ref int worstRef)
        => ScanCellsQ8Avx2Range(qPtr, cells, 0, cells.Length, offsets, candDist, candIdx, ref worstRef);

    private void ScanCellsQ8Avx2Range(
        sbyte* qPtr, ReadOnlySpan<int> cells, int ciStart, int ciEnd, int* offsets,
        Span<int> candDist, Span<int> candIdx, ref int worstRef)
    {
        if (_scalarAbort)
        {
            ScanCellsQ8ScalarAbortRange(qPtr, cells, ciStart, ciEnd, offsets, candDist, candIdx, ref worstRef);
            return;
        }
        var qLow = Vector128.Load(qPtr);
        var qWide = Vector256.WidenLower(qLow.ToVector256());
        var sbase = _dataset.Q8VectorsPtr;
        int worst = worstRef;

        for (int ci = ciStart; ci < ciEnd; ci++)
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
    /// REJECTED experiment (J3b): 2-row unrolled scan tentava processar 2 Q8 rows
    /// por iter via Vector256.Load(32B) + WidenLower/Upper para ganhar ILP.
    /// Bench mostrou regressão: p50 +10%, p99 +22%, final -85.
    /// Hipótese: o per-row branch (worst check) já é o gargalo dominante; dobrar o
    /// throughput de SIMD não ajuda quando insertion no top-K' precisa serializar.
    /// Código removido — ver plan.md "J3b" para análise completa.
    /// </summary>

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
    {        var sbase = _dataset.Q8VectorsPtr;
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

    /// <summary>
    /// J10: Scalar early-abort dim-by-dim per row. Inspired by the C #1 winner
    /// (thiagorigonatti/rinha-2026). For each row, accumulate squared diff one
    /// dim at a time and abort the row as soon as accumulator ≥ current worst.
    /// On a query whose top-K' worst is small, most rows abort within 2-4 dims,
    /// paying ~10 ops vs the AVX2 path's full 14-dim sqdiff (~30 ops total).
    /// Iff prune_rate &gt; ~70% across the workload, this beats AVX2.
    /// Uses fixed [0..13] dim order; selectivity-ordered would be a refinement.
    /// </summary>
    private void ScanCellsQ8ScalarAbortRange(
        sbyte* qPtr, ReadOnlySpan<int> cells, int ciStart, int ciEnd, int* offsets,
        Span<int> candDist, Span<int> candIdx, ref int worstRef)
    {
        var sbase = _dataset.Q8VectorsPtr;
        int worst = worstRef;
        // Hoist query dims into locals (registers) to avoid repeated loads.
        int q0 = qPtr[0], q1 = qPtr[1], q2 = qPtr[2], q3 = qPtr[3];
        int q4 = qPtr[4], q5 = qPtr[5], q6 = qPtr[6], q7 = qPtr[7];
        int q8 = qPtr[8], q9 = qPtr[9], q10 = qPtr[10], q11 = qPtr[11];
        int q12 = qPtr[12], q13 = qPtr[13];

        for (int ci = ciStart; ci < ciEnd; ci++)
        {
            int c = cells[ci];
            if (c < 0) break;
            int start = offsets[c];
            int end = offsets[c + 1];
            for (int i = start; i < end; i++)
            {
                sbyte* row = sbase + (long)i * PaddedDimensions;
                int d, acc;
                d = row[0] - q0;   acc  = d * d;
                d = row[1] - q1;   acc += d * d; if (acc >= worst) continue;
                d = row[2] - q2;   acc += d * d;
                d = row[3] - q3;   acc += d * d; if (acc >= worst) continue;
                d = row[4] - q4;   acc += d * d;
                d = row[5] - q5;   acc += d * d; if (acc >= worst) continue;
                d = row[6] - q6;   acc += d * d;
                d = row[7] - q7;   acc += d * d; if (acc >= worst) continue;
                d = row[8] - q8;   acc += d * d;
                d = row[9] - q9;   acc += d * d; if (acc >= worst) continue;
                d = row[10] - q10; acc += d * d;
                d = row[11] - q11; acc += d * d; if (acc >= worst) continue;
                d = row[12] - q12; acc += d * d;
                d = row[13] - q13; acc += d * d;
                if (acc < worst)
                {
                    InsertTopKInt(candDist, candIdx, acc, i);
                    worst = candDist[candDist.Length - 1];
                }
            }
        }
        worstRef = worst;
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
