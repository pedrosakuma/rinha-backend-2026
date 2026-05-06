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
    private readonly int _scalarAbort; // 0=off, 1=AoS dim-unroll (J10), 2=SoA survivor compaction (J11)
    private readonly bool _densityOrder; // J12a: reorder visited cells (within early-stop block) by size desc

    /// <summary>Per-thread last-query telemetry. 0=full scan, 1=early stop at early checkpoint,
    /// 2=early stop at main checkpoint. Updated after each Score() call.</summary>
    [ThreadStatic] public static int LastEarlyStopMode;
    /// <summary>Per-thread total Q8 rows scanned in the last query (sum of cell sizes scanned).</summary>
    [ThreadStatic] public static int LastRowsScanned;

    // ---------------- What-if instrumentation (PROFILE_TIMING=2) -----------------
    /// <summary>Candidate checkpoint percentages probed by ScoreWhatIf. Programmatically
    /// chosen to cover the empirically interesting band. Order matters for table output.</summary>
    public static readonly int[] WhatIfPcts = { 60, 70, 72, 75, 78, 80, 82, 85, 88, 90 };
    /// <summary>Per-thread, per-pct "would the gate have fired?" booleans (0 or 1) for the last query.</summary>
    [ThreadStatic] public static int[]? LastWhatIfPass;
    /// <summary>Per-thread, per-pct margin slack = cellsDist[chk] - bestDist[K-1]. Negative iff
    /// margin condition failed. Computed using the *unanimous* check; -∞ if not unanimous.</summary>
    [ThreadStatic] public static float[]? LastWhatIfSlack;
    /// <summary>Per-thread, per-pct unanimity flag (independent of margin).</summary>
    [ThreadStatic] public static int[]? LastWhatIfUnanimous;
    /// <summary>The smallest pct at which the gate would have fired (margin AND unanimous), or -1.</summary>
    [ThreadStatic] public static int LastWhatIfMinPassPct;
    /// <summary>Cell-visit counter shared across queries (atomic increments). Length = NumCells.</summary>
    public static int[]? CellVisits;

    /// <summary>J24: per-call deadline override in microseconds. When &gt; 0, takes precedence
    /// over the static IVF_DEADLINE_US. Set this from the request handler after running the
    /// hard-query classifier; reset to 0 after Score() returns. ThreadStatic, no contention.</summary>
    [ThreadStatic] public static int CallDeadlineUs;

    /// <summary>L4: per-call nProbe override. When &gt; 0 and ≤ constructor _nProbe, scans only
    /// the top-N cells instead of the full _nProbe. Used by the request handler to drop nProbe
    /// on easy queries (classifier-predicted), saving Q8 scan work on the bulk.</summary>
    [ThreadStatic] public static int CallNProbe;

    // J21: relax early-stop unanimity from 5/5 to 4/5 (or 1/5). Reduces tail-cell scans on
    // borderline queries at small cost in label accuracy. Env: IVF_EARLY_MAJORITY=1.
    private static readonly bool s_earlyMajority =
        Environment.GetEnvironmentVariable("IVF_EARLY_MAJORITY") == "1";

    // J22: anytime/deadline scan — when set, the *tail* Q8 scan (post-checkpoint, after
    // early-stop fails) cuts off after this many microseconds from query start. Returns
    // an approximate top-K based on cells visited so far. Env: IVF_DEADLINE_US (0 = off).
    private static readonly long s_deadlineTicks = ComputeDeadlineTicks();
    private static long ComputeDeadlineTicks()
    {
        var s = Environment.GetEnvironmentVariable("IVF_DEADLINE_US");
        if (string.IsNullOrEmpty(s) || !int.TryParse(s, out var us) || us <= 0) return 0;
        return us * System.Diagnostics.Stopwatch.Frequency / 1_000_000L;
    }

    public IvfScorer(Dataset dataset, int nProbe = 16, int kPrime = 32, int dimFilter = -1, int dimFilter2 = -1, bool earlyStop = false, int earlyStopPct = 75, bool bboxRepair = false, int earlyStopPctEarly = 0, int scalarAbort = 0, bool densityOrder = false)
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
        // Mode 2 (SoA) requires Q8-SoA file; silently downgrade if missing.
        if (scalarAbort == 2 && !dataset.HasQ8Soa) scalarAbort = 0;
        _scalarAbort = scalarAbort is >= 0 and <= 2 ? scalarAbort : 0;
        _densityOrder = densityOrder;
    }

    public float Score(ReadOnlySpan<float> query)
    {
        if (query.Length < Dataset.Dimensions)
            throw new ArgumentException("Query vector too small", nameof(query));

        LastEarlyStopMode = 0;
        LastRowsScanned = 0;
        long deadlineAt = 0;
        if (CallDeadlineUs > 0)
        {
            deadlineAt = System.Diagnostics.Stopwatch.GetTimestamp()
                + (long)CallDeadlineUs * System.Diagnostics.Stopwatch.Frequency / 1_000_000L;
        }
        else if (s_deadlineTicks > 0)
        {
            deadlineAt = System.Diagnostics.Stopwatch.GetTimestamp() + s_deadlineTicks;
        }

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
        // L4: per-call nProbe override. Buffer stays at _nProbe (max), but we only fill
        // & scan the top-n cells. Cells[n.._nProbe-1] stay sentinel -1 so scan loops break.
        int n = (CallNProbe > 0 && CallNProbe <= _nProbe) ? CallNProbe : _nProbe;
        Span<float> centDist = stackalloc float[512]; // safe upper bound; nlist ≤ 512 by config
        if (nlist > 512)
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

        // 3) Pick top-n cells (small partial sort; buffer is sized to _nProbe = max).
        Span<int>   cells     = stackalloc int[_nProbe];
        Span<float> cellsDist = stackalloc float[_nProbe];
        for (int i = 0; i < _nProbe; i++) { cells[i] = -1; cellsDist[i] = float.PositiveInfinity; }
        float cellsWorst = float.PositiveInfinity;
        for (int c = 0; c < nlist; c++)
        {
            float d = centDist[c];
            if (d < cellsWorst)
            {
                int pos = n - 1;
                while (pos > 0 && cellsDist[pos - 1] > d) pos--;
                for (int j = n - 1; j > pos; j--) { cellsDist[j] = cellsDist[j - 1]; cells[j] = cells[j - 1]; }
                cellsDist[pos] = d;
                cells[pos] = c;
                cellsWorst = cellsDist[n - 1];
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
                    int checkpoint = (n * _earlyStopPct) / 100;
                    int earlyCheckpoint = (n * _earlyStopPctEarly) / 100;
                    int curStart = 0;

                    // J12a: optionally reorder cells inside the pre-checkpoint block by
                    // size desc, so dense cells contribute candidates earliest. The gate
                    // invariant uses cellsDist[checkpoint] (a value, not an index relation),
                    // so permuting cells[0..checkpoint) leaves it intact.
                    if (_densityOrder && checkpoint > 1)
                    {
                        var offs = _dataset.CellOffsetsPtr;
                        // Insertion sort by descending cell size within [0..checkpoint).
                        for (int i = 1; i < checkpoint; i++)
                        {
                            int ci = cells[i];
                            int szi = (int)(offs[ci + 1] - offs[ci]);
                            int j = i - 1;
                            while (j >= 0)
                            {
                                int cj = cells[j];
                                int szj = (int)(offs[cj + 1] - offs[cj]);
                                if (szj >= szi) break;
                                cells[j + 1] = cj;
                                j--;
                            }
                            cells[j + 1] = ci;
                        }
                    }

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
                        bool unanimous0 = (valid0 == K) && (s_earlyMajority ? (frauds0 <= 1 || frauds0 >= K - 1) : (frauds0 == 0 || frauds0 == K));
                        bool marginOk0 = unanimous0 && bestDist[K - 1] < cellsDist[earlyCheckpoint];
                        if (marginOk0)
                        {
                            earlyStopped = true;
                            LastEarlyStopMode = 1;
                        }
                        else
                        {
                            curStart = earlyCheckpoint;
                        }
                    }

                    if (!earlyStopped && checkpoint >= 1 && n > checkpoint)
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
                        bool unanimous = (valid == K) && (s_earlyMajority ? (frauds <= 1 || frauds >= K - 1) : (frauds == 0 || frauds == K));
                        bool marginOk = unanimous && bestDist[K - 1] < cellsDist[checkpoint];
                        if (marginOk)
                        {
                            earlyStopped = true;
                            LastEarlyStopMode = 2;
                        }
                        else
                        {
                            ScanCellsQ8Avx2RangeDeadline(qPtr, cells, checkpoint, n, offsets, candDist, candIdx, ref q8Worst, deadlineAt);
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
    /// L1: batched scoring of 2 queries in a single sweep over the union of probe sets.
    /// Each row in the union is loaded once from L1 and produces TWO Q8 distances in
    /// parallel (one per query) — amortizes memory bandwidth across queries.
    /// No early-stop / no bbox repair / no per-call deadline (simplification for MVP).
    /// </summary>
    public void ScoreBatch2(ReadOnlySpan<float> q1raw, ReadOnlySpan<float> q2raw, out float s1, out float s2)
    {
        if (!Avx2.IsSupported)
        {
            s1 = Score(q1raw);
            s2 = Score(q2raw);
            return;
        }

        Span<sbyte> q1Q8 = stackalloc sbyte[PaddedDimensions];
        Span<sbyte> q2Q8 = stackalloc sbyte[PaddedDimensions];
        Span<float> q1Padded = stackalloc float[PaddedDimensions];
        Span<float> q2Padded = stackalloc float[PaddedDimensions];
        QuantizeAndPad(q1raw, q1Q8, q1Padded);
        QuantizeAndPad(q2raw, q2Q8, q2Padded);

        int nlist = _dataset.NumCells;
        Span<float> centDist1 = stackalloc float[512];
        Span<float> centDist2 = stackalloc float[512];

        fixed (float* qfPtr1 = q1Padded, qfPtr2 = q2Padded)
        {
            ComputeCentroidDistances(qfPtr1, centDist1, nlist);
            ComputeCentroidDistances(qfPtr2, centDist2, nlist);
        }

        Span<int> cells1 = stackalloc int[_nProbe];
        Span<int> cells2 = stackalloc int[_nProbe];
        Span<float> cellsDist1 = stackalloc float[_nProbe];
        Span<float> cellsDist2 = stackalloc float[_nProbe];
        PickTopCells(centDist1, nlist, cells1, cellsDist1);
        PickTopCells(centDist2, nlist, cells2, cellsDist2);

        Span<byte> cellMask = stackalloc byte[nlist];
        cellMask.Clear();
        for (int i = 0; i < _nProbe; i++) { int c = cells1[i]; if (c >= 0) cellMask[c] |= 1; }
        for (int i = 0; i < _nProbe; i++) { int c = cells2[i]; if (c >= 0) cellMask[c] |= 2; }

        Span<int> cand1Dist = stackalloc int[_kPrime];
        Span<int> cand1Idx  = stackalloc int[_kPrime];
        Span<int> cand2Dist = stackalloc int[_kPrime];
        Span<int> cand2Idx  = stackalloc int[_kPrime];
        for (int i = 0; i < _kPrime; i++)
        {
            cand1Dist[i] = int.MaxValue; cand1Idx[i] = -1;
            cand2Dist[i] = int.MaxValue; cand2Idx[i] = -1;
        }
        int worst1 = int.MaxValue, worst2 = int.MaxValue;

        var offsets = _dataset.CellOffsetsPtr;
        var sbase = _dataset.Q8VectorsPtr;
        long rowsScanned = 0;
        fixed (sbyte* qPtr1 = q1Q8, qPtr2 = q2Q8)
        {
            var qLow1 = Vector128.Load(qPtr1);
            var qLow2 = Vector128.Load(qPtr2);
            var qWide1 = Vector256.WidenLower(qLow1.ToVector256());
            var qWide2 = Vector256.WidenLower(qLow2.ToVector256());

            for (int c = 0; c < nlist; c++)
            {
                byte mask = cellMask[c];
                if (mask == 0) continue;
                int start = offsets[c];
                int end = offsets[c + 1];
                rowsScanned += end - start;

                if (mask == 3)
                {
                    for (int i = start; i < end; i++)
                    {
                        sbyte* row = sbase + (long)i * PaddedDimensions;
                        var r128 = Vector128.Load(row);
                        var rWide = Vector256.WidenLower(r128.ToVector256());
                        var diff1 = rWide - qWide1;
                        var diff2 = rWide - qWide2;
                        int d1 = Vector256.Sum(Avx2.MultiplyAddAdjacent(diff1, diff1));
                        int d2 = Vector256.Sum(Avx2.MultiplyAddAdjacent(diff2, diff2));
                        if (d1 < worst1) { InsertTopKInt(cand1Dist, cand1Idx, d1, i); worst1 = cand1Dist[_kPrime - 1]; }
                        if (d2 < worst2) { InsertTopKInt(cand2Dist, cand2Idx, d2, i); worst2 = cand2Dist[_kPrime - 1]; }
                    }
                }
                else if (mask == 1)
                {
                    for (int i = start; i < end; i++)
                    {
                        sbyte* row = sbase + (long)i * PaddedDimensions;
                        var r128 = Vector128.Load(row);
                        var rWide = Vector256.WidenLower(r128.ToVector256());
                        var diff = rWide - qWide1;
                        int d = Vector256.Sum(Avx2.MultiplyAddAdjacent(diff, diff));
                        if (d < worst1) { InsertTopKInt(cand1Dist, cand1Idx, d, i); worst1 = cand1Dist[_kPrime - 1]; }
                    }
                }
                else
                {
                    for (int i = start; i < end; i++)
                    {
                        sbyte* row = sbase + (long)i * PaddedDimensions;
                        var r128 = Vector128.Load(row);
                        var rWide = Vector256.WidenLower(r128.ToVector256());
                        var diff = rWide - qWide2;
                        int d = Vector256.Sum(Avx2.MultiplyAddAdjacent(diff, diff));
                        if (d < worst2) { InsertTopKInt(cand2Dist, cand2Idx, d, i); worst2 = cand2Dist[_kPrime - 1]; }
                    }
                }
            }
        }
        LastRowsScanned += (int)rowsScanned;

        Span<float> bestDist1 = stackalloc float[K];
        Span<int>   bestIdx1  = stackalloc int[K];
        Span<float> bestDist2 = stackalloc float[K];
        Span<int>   bestIdx2  = stackalloc int[K];
        for (int i = 0; i < K; i++)
        {
            bestDist1[i] = float.PositiveInfinity; bestIdx1[i] = -1;
            bestDist2[i] = float.PositiveInfinity; bestIdx2[i] = -1;
        }
        RerankFloat(q1Padded, cand1Dist, cand1Idx, bestDist1, bestIdx1);
        RerankFloat(q2Padded, cand2Dist, cand2Idx, bestDist2, bestIdx2);

        var labels = _dataset.LabelsPtr;
        int frauds1 = 0, frauds2 = 0;
        for (int i = 0; i < K; i++)
        {
            if (bestIdx1[i] >= 0 && labels[bestIdx1[i]] != 0) frauds1++;
            if (bestIdx2[i] >= 0 && labels[bestIdx2[i]] != 0) frauds2++;
        }
        s1 = frauds1 / (float)K;
        s2 = frauds2 / (float)K;
    }

    private static void QuantizeAndPad(ReadOnlySpan<float> raw, Span<sbyte> q8, Span<float> padded)
    {
        for (int i = 0; i < Dataset.Dimensions; i++)
        {
            padded[i] = raw[i];
            int q = (int)MathF.Round(raw[i] * Dataset.Q8Scale);
            if (q > 127) q = 127;
            else if (q < -128) q = -128;
            q8[i] = (sbyte)q;
        }
    }

    private void ComputeCentroidDistances(float* qfPtr, Span<float> centDist, int nlist)
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

    private void PickTopCells(ReadOnlySpan<float> centDist, int nlist, Span<int> cells, Span<float> cellsDist)
    {
        for (int i = 0; i < _nProbe; i++) { cells[i] = -1; cellsDist[i] = float.PositiveInfinity; }
        float worst = float.PositiveInfinity;
        for (int c = 0; c < nlist; c++)
        {
            float d = centDist[c];
            if (d < worst)
            {
                int pos = _nProbe - 1;
                while (pos > 0 && cellsDist[pos - 1] > d) pos--;
                for (int j = _nProbe - 1; j > pos; j--) { cellsDist[j] = cellsDist[j - 1]; cells[j] = cells[j - 1]; }
                cellsDist[pos] = d;
                cells[pos] = c;
                worst = cellsDist[_nProbe - 1];
            }
        }
    }

    /// Used by the cascade pre-classifier (see <see cref="Cascade"/>): it needs the same
    /// centroid distances the IvfScorer would compute, but stops there. Roughly ~2µs per
    /// query (one AVX2 dot-product per centroid, NLIST≤256).
    /// </summary>
    public void ComputeTop3Cells(
        ReadOnlySpan<float> query,
        Span<int> top3Cells,
        Span<float> top3Dists)
    {
        if (query.Length < Dataset.Dimensions)
            throw new ArgumentException("Query vector too small", nameof(query));
        if (top3Cells.Length < 3 || top3Dists.Length < 3)
            throw new ArgumentException("top3 spans must hold at least 3 elements");

        Span<float> paddedQuery = stackalloc float[PaddedDimensions];
        for (int i = 0; i < Dataset.Dimensions; i++) paddedQuery[i] = query[i];

        int nlist = _dataset.NumCells;
        // Maintain top-3 in ascending order: top3Dists[0] is smallest.
        top3Dists[0] = top3Dists[1] = top3Dists[2] = float.PositiveInfinity;
        top3Cells[0] = top3Cells[1] = top3Cells[2] = -1;

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
                var dlo = k0 - q0;
                var dhi = k1 - q1;
                float d = Vector256.Sum((dlo * dlo) + (dhi * dhi));
                if (d < top3Dists[2])
                {
                    if (d < top3Dists[1])
                    {
                        top3Dists[2] = top3Dists[1]; top3Cells[2] = top3Cells[1];
                        if (d < top3Dists[0])
                        {
                            top3Dists[1] = top3Dists[0]; top3Cells[1] = top3Cells[0];
                            top3Dists[0] = d;            top3Cells[0] = c;
                        }
                        else
                        {
                            top3Dists[1] = d; top3Cells[1] = c;
                        }
                    }
                    else
                    {
                        top3Dists[2] = d; top3Cells[2] = c;
                    }
                }
            }
        }
    }

    /// <summary>
    /// What-if instrumented scoring path. Always full-scan (no early-stop), but at each
    /// candidate checkpoint percentage in <see cref="WhatIfPcts"/> snapshots:
    /// (a) would the unanimity gate hold, (b) would the margin condition
    /// bestDist[K-1] &lt; cellsDist[chk] hold, (c) the slack value cellsDist[chk]-bestDist[K-1].
    ///
    /// Result is identical to a no-early-stop Score() call. Per-pct telemetry written into
    /// the [ThreadStatic] LastWhatIf* arrays for the caller to aggregate.
    ///
    /// Cost: O(num_checkpoints * kPrime) extra rerank work per query (~few µs total). Not
    /// for production use — gated by Program.cs PROFILE_TIMING=2.
    /// </summary>
    public float ScoreWithWhatIf(ReadOnlySpan<float> query)
    {
        if (query.Length < Dataset.Dimensions)
            throw new ArgumentException("Query vector too small", nameof(query));

        int npc = WhatIfPcts.Length;
        var pass = LastWhatIfPass;
        if (pass is null || pass.Length != npc) pass = LastWhatIfPass = new int[npc];
        var slack = LastWhatIfSlack;
        if (slack is null || slack.Length != npc) slack = LastWhatIfSlack = new float[npc];
        var unanimous = LastWhatIfUnanimous;
        if (unanimous is null || unanimous.Length != npc) unanimous = LastWhatIfUnanimous = new int[npc];
        for (int i = 0; i < npc; i++) { pass[i] = 0; slack[i] = float.NegativeInfinity; unanimous[i] = 0; }
        LastWhatIfMinPassPct = -1;

        // 1) Quantize query.
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
        Span<float> centDist = stackalloc float[512];
        if (nlist > 512) centDist = new float[nlist];

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

        // 3) Top-NPROBE cells.
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

        // Pre-compute the cell index of each pct boundary (clamped to [1, nProbe]).
        Span<int> pctChk = stackalloc int[npc];
        for (int p = 0; p < npc; p++)
        {
            int chk = (_nProbe * WhatIfPcts[p]) / 100;
            if (chk < 1) chk = 1;
            if (chk > _nProbe) chk = _nProbe;
            pctChk[p] = chk;
        }

        // 4) Cell-visit counter (atomic, optional).
        var visits = CellVisits;
        if (visits is not null)
        {
            for (int i = 0; i < _nProbe; i++)
            {
                int ci = cells[i];
                if ((uint)ci < (uint)visits.Length) System.Threading.Interlocked.Increment(ref visits[ci]);
            }
        }

        // 5) Scan cells in order, snapshotting at each pct checkpoint.
        Span<int> candDist = stackalloc int[_kPrime];
        Span<int> candIdx  = stackalloc int[_kPrime];
        for (int i = 0; i < _kPrime; i++) { candDist[i] = int.MaxValue; candIdx[i] = -1; }
        int q8Worst = int.MaxValue;
        Span<float> bestDist = stackalloc float[K];
        Span<int>   bestIdx  = stackalloc int[K];

        var offsets = _dataset.CellOffsetsPtr;
        var labels  = _dataset.LabelsPtr;
        int prevChk = 0;
        int minPassPct = -1;
        fixed (sbyte* qPtr = qQ8)
        {
            for (int p = 0; p < npc; p++)
            {
                int chk = pctChk[p];
                if (chk > prevChk)
                {
                    if (Avx2.IsSupported)
                        ScanCellsQ8Avx2Range(qPtr, cells, prevChk, chk, offsets, candDist, candIdx, ref q8Worst);
                    else
                        ScanCellsQ8Scalar(qPtr, cells, offsets, candDist, candIdx, ref q8Worst);
                    prevChk = chk;
                }
                // Snapshot rerank fresh from current cands.
                for (int i = 0; i < K; i++) { bestDist[i] = float.PositiveInfinity; bestIdx[i] = -1; }
                RerankFloat(paddedQuery, candDist, candIdx, bestDist, bestIdx);
                int valid = 0, frauds = 0;
                for (int i = 0; i < K; i++)
                {
                    if (bestIdx[i] < 0) continue;
                    valid++;
                    if (labels[bestIdx[i]] != 0) frauds++;
                }
                bool isUnanimous = (valid == K) && (frauds == 0 || frauds == K);
                float chkDist = chk < _nProbe ? cellsDist[chk] : float.PositiveInfinity;
                float worstD  = bestDist[K - 1];
                float sl = float.IsPositiveInfinity(worstD) ? float.NegativeInfinity : (chkDist - worstD);
                bool marginOk = isUnanimous && worstD < chkDist;
                unanimous[p] = isUnanimous ? 1 : 0;
                slack[p] = sl;
                pass[p] = marginOk ? 1 : 0;
                if (marginOk && minPassPct < 0) minPassPct = WhatIfPcts[p];
            }
            // Drain any remaining cells (last pct may be < 100).
            if (prevChk < _nProbe)
            {
                if (Avx2.IsSupported)
                    ScanCellsQ8Avx2Range(qPtr, cells, prevChk, _nProbe, offsets, candDist, candIdx, ref q8Worst);
                else
                    ScanCellsQ8Scalar(qPtr, cells, offsets, candDist, candIdx, ref q8Worst);
                for (int i = 0; i < K; i++) { bestDist[i] = float.PositiveInfinity; bestIdx[i] = -1; }
                RerankFloat(paddedQuery, candDist, candIdx, bestDist, bestIdx);
            }
        }

        LastWhatIfMinPassPct = minPassPct;

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

    private void ScanCellsQ8Avx2RangeDeadline(
        sbyte* qPtr, ReadOnlySpan<int> cells, int ciStart, int ciEnd, int* offsets,
        Span<int> candDist, Span<int> candIdx, ref int worstRef, long deadlineAt)
    {
        if (deadlineAt == 0 || _scalarAbort != 0)
        {
            ScanCellsQ8Avx2Range(qPtr, cells, ciStart, ciEnd, offsets, candDist, candIdx, ref worstRef);
            return;
        }
        var qLow = Vector128.Load(qPtr);
        var qWide = Vector256.WidenLower(qLow.ToVector256());
        var sbase = _dataset.Q8VectorsPtr;
        int worst = worstRef;
        int rowsScanned = 0;
        for (int ci = ciStart; ci < ciEnd; ci++)
        {
            int c = cells[ci];
            if (c < 0) break;
            // Deadline check at cell boundary (~µs granularity, cheap RDTSC).
            if (System.Diagnostics.Stopwatch.GetTimestamp() > deadlineAt) break;
            int start = offsets[c];
            int end = offsets[c + 1];
            rowsScanned += end - start;
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
        LastRowsScanned += rowsScanned;
    }

    private void ScanCellsQ8Avx2Range(
        sbyte* qPtr, ReadOnlySpan<int> cells, int ciStart, int ciEnd, int* offsets,
        Span<int> candDist, Span<int> candIdx, ref int worstRef)
    {
        if (_scalarAbort == 1)
        {
            ScanCellsQ8ScalarAbortRange(qPtr, cells, ciStart, ciEnd, offsets, candDist, candIdx, ref worstRef);
            return;
        }
        if (_scalarAbort == 2)
        {
            ScanCellsQ8SoaAbortRange(qPtr, cells, ciStart, ciEnd, offsets, candDist, candIdx, ref worstRef);
            return;
        }
        var qLow = Vector128.Load(qPtr);
        var qWide = Vector256.WidenLower(qLow.ToVector256());
        var sbase = _dataset.Q8VectorsPtr;
        int worst = worstRef;
        int rowsScanned = 0;

        for (int ci = ciStart; ci < ciEnd; ci++)
        {
            int c = cells[ci];
            if (c < 0) break;
            int start = offsets[c];
            int end = offsets[c + 1];
            rowsScanned += end - start;
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
        LastRowsScanned += rowsScanned;
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
    /// J11: SoA scalar early-abort with row-survivor compaction (the C #1 winner pattern).
    /// Per cluster:
    ///   pass 0: stream dim 0 sequentially over all rows → compute partial acc, build survivor[].
    ///   pass d (1..13): for each survivor, accumulate dim d's sqdiff; if acc &gt;= worst, drop.
    /// Sequential per-dim streaming = perfect prefetch; survivor list shrinks geometrically
    /// so total work is roughly N + N×p + N×p² + ... ≈ N/(1-p), much less than 14×N for high prune rate.
    /// </summary>
    private void ScanCellsQ8SoaAbortRange(
        sbyte* qPtr, ReadOnlySpan<int> cells, int ciStart, int ciEnd, int* offsets,
        Span<int> candDist, Span<int> candIdx, ref int worstRef)
    {
        var soa = _dataset.Q8SoaPtr;
        long N = _dataset.Count;
        int worst = worstRef;

        // Hoist query dims.
        int q0 = qPtr[0], q1 = qPtr[1], q2 = qPtr[2], q3 = qPtr[3];
        int q4 = qPtr[4], q5 = qPtr[5], q6 = qPtr[6], q7 = qPtr[7];
        int q8 = qPtr[8], q9 = qPtr[9], q10 = qPtr[10], q11 = qPtr[11];
        int q12 = qPtr[12], q13 = qPtr[13];

        // Per-dim base pointers in the SoA file.
        sbyte* p0 = soa + 0 * N, p1 = soa + 1 * N, p2 = soa + 2 * N, p3 = soa + 3 * N;
        sbyte* p4 = soa + 4 * N, p5 = soa + 5 * N, p6 = soa + 6 * N, p7 = soa + 7 * N;
        sbyte* p8 = soa + 8 * N, p9 = soa + 9 * N, p10 = soa + 10 * N, p11 = soa + 11 * N;
        sbyte* p12 = soa + 12 * N, p13 = soa + 13 * N;

        const int MaxCellSize = 65536; // generous cap; nlist=256 over 3M ≈ 12K avg
        Span<int> acc = stackalloc int[MaxCellSize];
        Span<int> surv = stackalloc int[MaxCellSize];

        for (int ci = ciStart; ci < ciEnd; ci++)
        {
            int c = cells[ci];
            if (c < 0) break;
            int start = offsets[c];
            int end = offsets[c + 1];
            int sz = end - start;
            if (sz <= 0) continue;
            if (sz > MaxCellSize) sz = MaxCellSize; // safety

            // Pass 0: dim 0, all rows. acc[r] = (p0[start+r] - q0)^2; if < worst, mark survivor.
            int n = 0;
            for (int r = 0; r < sz; r++)
            {
                int d = p0[start + r] - q0;
                int a = d * d;
                acc[r] = a;
                if (a < worst) surv[n++] = r;
            }

            // Pass d (1..13): only iterate survivors; further compact.
            n = AddDimAndCompact(p1, start, q1, acc, surv, n, worst);
            n = AddDimAndCompact(p2, start, q2, acc, surv, n, worst);
            n = AddDimAndCompact(p3, start, q3, acc, surv, n, worst);
            n = AddDimAndCompact(p4, start, q4, acc, surv, n, worst);
            n = AddDimAndCompact(p5, start, q5, acc, surv, n, worst);
            n = AddDimAndCompact(p6, start, q6, acc, surv, n, worst);
            n = AddDimAndCompact(p7, start, q7, acc, surv, n, worst);
            n = AddDimAndCompact(p8, start, q8, acc, surv, n, worst);
            n = AddDimAndCompact(p9, start, q9, acc, surv, n, worst);
            n = AddDimAndCompact(p10, start, q10, acc, surv, n, worst);
            n = AddDimAndCompact(p11, start, q11, acc, surv, n, worst);
            n = AddDimAndCompact(p12, start, q12, acc, surv, n, worst);
            n = AddDimAndCompact(p13, start, q13, acc, surv, n, worst);

            // Insert remaining survivors into top-K'.
            for (int s = 0; s < n; s++)
            {
                int r = surv[s];
                int dist = acc[r];
                if (dist < worst)
                {
                    InsertTopKInt(candDist, candIdx, dist, start + r);
                    worst = candDist[candDist.Length - 1];
                }
            }
        }
        worstRef = worst;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int AddDimAndCompact(sbyte* dimBase, int start, int qd, Span<int> acc, Span<int> surv, int nIn, int worst)
    {
        int nOut = 0;
        for (int s = 0; s < nIn; s++)
        {
            int r = surv[s];
            int d = dimBase[start + r] - qd;
            int a = acc[r] + d * d;
            acc[r] = a;
            if (a < worst) surv[nOut++] = r;
        }
        return nOut;
    }

    /// <summary>
    /// J10: Scalar early-abort dim-by-dim per row (AoS layout). Inspired by the C #1 winner
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
