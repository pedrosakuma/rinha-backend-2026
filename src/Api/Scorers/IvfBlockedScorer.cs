using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Rinha.Api.Scorers;

/// <summary>
/// Block-SoA Q16 IVF scorer with **exact** pruning (recall-equivalent to brute-force top-K).
///
/// Algorithm — the "best of two worlds" port of jairoblatt/whereisanzi's scan_blocks:
///   1) Compute squared L2 to all centroids (AoS Vector256, 16 floats per centroid).
///   2) Pick FAST_NPROBE clusters by smallest centroid distance and scan their blocks.
///   3) bbox-LB pass over remaining clusters: for each cluster c not in the FAST set,
///      lb(c, q) = Σ_d max(0, max(bmin[c,d] - q[d], q[d] - bmax[c,d]))² is a provable
///      lower bound on min dist² in c. Skip iff lb(c, q) > top.worst. Visit otherwise.
///   4) For each block (8 reference vectors): compute first 8 dims (4 fmadd pairs),
///      partial-check vs current top-5 worst; if no lane &lt; worst, skip block.
///      Otherwise finish the remaining 6 dims and update top-5 via worst-idx tracking
///      (replace worst, linear-scan 5 elems to find new worst).
///
/// Both prunes are **admissible** (lower bounds; never exclude a possible top-K member),
/// so this delivers identical recall to brute-force regardless of input distribution.
/// The cascade gate from jairoblatt's original (`if count ∈ {2,3} only escalate`) is
/// **omitted by default** — it's the only step that's heuristic and OOD-fragile.
///
/// Hardware optimizations from jairoblatt's reference:
///   - VPMADDWD-style FMA in float32: cvtepi16→cvtepi32→cvtepi32_ps × scale, fmadd diff².
///   - Software prefetch T0 of block_i+8 (head + midpoint = 2 cache lines of 4).
///   - Top-5 worst-idx tracking instead of sorted insert.
///   - bbox precomputed in i16/Q16 units (eliminates per-call MathF.Round in hot path) —
///     stored as float32 because the bbox-LB is computed in float32 (matches the q[d]
///     query passed to FMA) and avoids any quant noise in the LB itself.
///
/// Env: IVF_BLOCKED_NPROBE (default 4), IVF_BLOCKED_FAST_GATE=1 to enable the OOD-fragile
/// cascade gate (default off — exact pass).
/// </summary>
public sealed unsafe class IvfBlockedScorer : IFraudScorer
{
    private const int K = 5;
    private const int Dimensions = 14;
    private const int PaddedDimensions = 16;
    private const int BlockSize = 8;
    private const int BlockShorts = BlockSize * Dimensions; // 112 i16/block
    private const int BlockBytes = BlockShorts * sizeof(short); // 224 B/block

    private readonly Dataset _dataset;
    private readonly int _nProbe;
    private readonly bool _fastGate; // OOD-fragile cascade (jairoblatt's behaviour); default off
    private readonly bool _hasBbox;

    public IvfBlockedScorer(Dataset dataset, int nProbe = 4)
    {
        if (!dataset.HasIvf) throw new InvalidOperationException("Dataset has no IVF (centroids/offsets) view.");
        if (!dataset.HasQ16Blocked) throw new InvalidOperationException("Dataset has no Q16-blocked view.");
        _dataset = dataset;
        _nProbe = Math.Clamp(nProbe, 1, dataset.NumCells);
        _fastGate = Environment.GetEnvironmentVariable("IVF_BLOCKED_FAST_GATE") == "1";
        _hasBbox = dataset.HasIvfBbox;
    }

    public float Score(ReadOnlySpan<float> query)
    {
        if (query.Length < Dimensions)
            throw new ArgumentException("Query vector too small", nameof(query));

        int nlist = _dataset.NumCells;
        Span<float> centDist = stackalloc float[512];
        if (nlist > 512) centDist = new float[nlist];
        Span<float> qfPad = stackalloc float[PaddedDimensions];
        for (int i = 0; i < Dimensions; i++) qfPad[i] = query[i];

        // 1) Centroid distances (AoS centroids: 16 floats per centroid).
        fixed (float* qPtr = qfPad)
        {
            var q0 = Vector256.Load(qPtr);
            var q1 = Vector256.Load(qPtr + 8);
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

        // 2) Top-N FAST cells.
        int n = _nProbe;
        Span<int> cells = stackalloc int[n];
        Span<float> cellsDist = stackalloc float[n];
        for (int i = 0; i < n; i++) { cells[i] = -1; cellsDist[i] = float.PositiveInfinity; }
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

        // 3) Top-5 (dist², label) tracked via worst-idx (linear scan of 5 elems on insert).
        // Slot 0..K-1; topDist[worstIdx] is the threshold compared to partial sums.
        Span<float> topDist = stackalloc float[K];
        Span<byte> topLab = stackalloc byte[K];
        Span<int> topIdx = stackalloc int[K]; // tiebreak by smaller orig_id (post-IVF reorder index)
        for (int i = 0; i < K; i++) { topDist[i] = float.PositiveInfinity; topLab[i] = 0; topIdx[i] = int.MaxValue; }
        int worstIdx = 0;

        // Broadcast each query dim once for fmadd reuse.
        Span<Vector256<float>> qVecs = stackalloc Vector256<float>[Dimensions];
        for (int d = 0; d < Dimensions; d++) qVecs[d] = Vector256.Create(query[d]);

        var blocksPtr = _dataset.Q16BlockedPtr;
        var blockOffs = _dataset.BlockOffsetsPtr;
        var labelsPtr = _dataset.LabelsPtr;
        var cellOffsPtr = _dataset.CellOffsetsPtr;

        // Scan FAST cells.
        for (int i = 0; i < n; i++)
        {
            int c = cells[i];
            if (c < 0) continue;
            ScanCellBlocks(c, blocksPtr, blockOffs, labelsPtr, cellOffsPtr, qVecs,
                topDist, topLab, topIdx, ref worstIdx);
        }

        // 4) Exact bbox-LB pass over remaining cells (when bbox available and fast-gate disabled).
        // Mathematically equivalent to brute-force top-K: every cluster whose lower bound exceeds
        // the current worst-top-5 cannot contain a closer neighbour and is provably safe to skip.
        if (_hasBbox && !_fastGate)
        {
            // Build seed mask in stack (small): nlist ≤ 512 by config.
            Span<bool> isSeed = stackalloc bool[nlist];
            for (int i = 0; i < n; i++) { int sc = cells[i]; if (sc >= 0) isSeed[sc] = true; }
            float* bbMin = _dataset.IvfBboxMinPtr;
            float* bbMax = _dataset.IvfBboxMaxPtr;

            fixed (float* qPtr = qfPad)
            {
                for (int c = 0; c < nlist; c++)
                {
                    if (isSeed[c]) continue;
                    float worst = topDist[worstIdx];
                    float lb = BboxLowerBoundSquared(qPtr, bbMin, bbMax, c);
                    if (lb >= worst) continue; // exact: cluster cannot beat top-5
                    ScanCellBlocks(c, blocksPtr, blockOffs, labelsPtr, cellOffsPtr, qVecs,
                        topDist, topLab, topIdx, ref worstIdx);
                }
            }
        }
        else if (_fastGate)
        {
            // Heuristic cascade (jairoblatt original): only escalate when count ∈ {2,3}.
            // Default OFF — kept for A/B vs the exact pass.
            int frauds0 = 0;
            for (int i = 0; i < K; i++) if (topLab[i] != 0) frauds0++;
            if (frauds0 == 2 || frauds0 == 3)
            {
                // Expand to FULL (3× FAST as in reference). Picks more cells from centDist.
                int nFull = Math.Min(_nProbe * 3, nlist);
                Span<int> fullCells = stackalloc int[nFull];
                Span<float> fullCellsDist = stackalloc float[nFull];
                for (int i = 0; i < nFull; i++) { fullCells[i] = -1; fullCellsDist[i] = float.PositiveInfinity; }
                float fw = float.PositiveInfinity;
                for (int c = 0; c < nlist; c++)
                {
                    float d = centDist[c];
                    if (d < fw)
                    {
                        int pos = nFull - 1;
                        while (pos > 0 && fullCellsDist[pos - 1] > d) pos--;
                        for (int j = nFull - 1; j > pos; j--) { fullCellsDist[j] = fullCellsDist[j - 1]; fullCells[j] = fullCells[j - 1]; }
                        fullCellsDist[pos] = d;
                        fullCells[pos] = c;
                        fw = fullCellsDist[nFull - 1];
                    }
                }
                Span<bool> visitedF = stackalloc bool[nlist];
                for (int i = 0; i < n; i++) { int sc = cells[i]; if (sc >= 0) visitedF[sc] = true; }
                for (int i = 0; i < nFull; i++)
                {
                    int c = fullCells[i];
                    if (c < 0 || visitedF[c]) continue;
                    visitedF[c] = true;
                    ScanCellBlocks(c, blocksPtr, blockOffs, labelsPtr, cellOffsPtr, qVecs,
                        topDist, topLab, topIdx, ref worstIdx);
                }
            }
        }

        int totalFrauds = 0;
        for (int i = 0; i < K; i++) if (topLab[i] != 0) totalFrauds++;
        return totalFrauds / (float)K;
    }

    /// <summary>Scans all blocks of a single cell, updating top-5 in place.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void ScanCellBlocks(int cell,
        short* blocksPtr, int* blockOffs, byte* labelsPtr, int* cellOffsPtr,
        Span<Vector256<float>> qVecs,
        Span<float> topDist, Span<byte> topLab, Span<int> topIdx,
        ref int worstIdx)
    {
        int blockStart = blockOffs[cell];
        int blockEnd = blockOffs[cell + 1];
        int rowStart = cellOffsPtr[cell]; // first row of cell (matches block_i*8 offset within cell)

        var scale = Vector256.Create(1f / Dataset.Q16Scale); // 1e-4
        Span<float> distsBuf = stackalloc float[8];

        for (int blockI = blockStart; blockI < blockEnd; blockI++)
        {
            // Software prefetch the +8 block (2 lines covering head + midpoint of 224B).
            int prefetchBlock = blockI + 8;
            if (prefetchBlock < blockEnd)
            {
                short* pp = blocksPtr + (long)prefetchBlock * BlockShorts;
                Sse.Prefetch0(pp);
                Sse.Prefetch0(pp + 56); // 56 shorts = 112 bytes in
            }

            short* bp = blocksPtr + (long)blockI * BlockShorts;
            float worst = topDist[worstIdx];
            var threshold = Vector256.Create(worst);

            // Process dims in pairs (acc0 / acc1) to break dependency chains.
            var acc0 = Vector256<float>.Zero;
            var acc1 = Vector256<float>.Zero;

            // First 4 pairs = 8 dims. After this we check for early-abort.
            DimPair(bp, 0, qVecs, scale, ref acc0, ref acc1);
            DimPair(bp, 2, qVecs, scale, ref acc0, ref acc1);
            DimPair(bp, 4, qVecs, scale, ref acc0, ref acc1);
            DimPair(bp, 6, qVecs, scale, ref acc0, ref acc1);

            var partial = acc0 + acc1;
            // Partial L2² is monotonic non-decreasing → if all lanes ≥ worst, full ≥ worst, skip block.
            int partialMask = (int)Vector256.LessThan(partial, threshold).ExtractMostSignificantBits();
            if (partialMask == 0) continue;

            // Remaining 3 pairs = 6 dims (8..13).
            DimPair(bp, 8, qVecs, scale, ref acc0, ref acc1);
            DimPair(bp, 10, qVecs, scale, ref acc0, ref acc1);
            DimPair(bp, 12, qVecs, scale, ref acc0, ref acc1);

            var full = acc0 + acc1;
            int mask = (int)Vector256.LessThan(full, threshold).ExtractMostSignificantBits();
            if (mask == 0) continue;

            // Extract candidate distances (only those passing mask) and merge into top-5.
            full.CopyTo(distsBuf);
            int labelBase = rowStart + (blockI - blockStart) * BlockSize;
            int m = mask;
            while (m != 0)
            {
                int slot = System.Numerics.BitOperations.TrailingZeroCount(m);
                m &= m - 1;
                float di = distsBuf[slot];
                int origIdx = labelBase + slot;
                // Re-fetch worst (may have been updated this iteration).
                float curWorst = topDist[worstIdx];
                if (di < curWorst || (di == curWorst && origIdx < topIdx[worstIdx]))
                {
                    topDist[worstIdx] = di;
                    topLab[worstIdx] = labelsPtr[origIdx];
                    topIdx[worstIdx] = origIdx;
                    // Recompute worst-idx via linear scan of 5 slots.
                    int wi = 0;
                    float wv = topDist[0];
                    int wIdxOrig = topIdx[0];
                    for (int j = 1; j < K; j++)
                    {
                        float v = topDist[j];
                        int io = topIdx[j];
                        // worst = larger dist; tiebreak by larger orig_id (so we evict it last).
                        if (v > wv || (v == wv && io > wIdxOrig)) { wv = v; wi = j; wIdxOrig = io; }
                    }
                    worstIdx = wi;
                }
            }
        }
    }

    /// <summary>Computes (v_d - q_d)² and (v_{d+1} - q_{d+1})² for 8 lanes, fmadd into acc0/acc1.
    /// bp is the base of the block; dim d's lane row is at bp + d*8.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void DimPair(short* bp, int d,
        Span<Vector256<float>> qVecs, Vector256<float> scale,
        ref Vector256<float> acc0, ref Vector256<float> acc1)
    {
        var r0 = Sse2.LoadVector128((short*)(bp + d * BlockSize));
        var r1 = Sse2.LoadVector128((short*)(bp + (d + 1) * BlockSize));
        var v0 = Avx2.ConvertToVector256Single(Avx2.ConvertToVector256Int32(r0)) * scale;
        var v1 = Avx2.ConvertToVector256Single(Avx2.ConvertToVector256Int32(r1)) * scale;
        var diff0 = v0 - qVecs[d];
        var diff1 = v1 - qVecs[d + 1];
        // Use FMA for diff*diff + acc when available; falls back to separate mul+add.
        if (Fma.IsSupported)
        {
            acc0 = Fma.MultiplyAdd(diff0, diff0, acc0);
            acc1 = Fma.MultiplyAdd(diff1, diff1, acc1);
        }
        else
        {
            acc0 += diff0 * diff0;
            acc1 += diff1 * diff1;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float BboxLowerBoundSquared(float* q, float* bbMin, float* bbMax, int cell)
    {
        float* mn = bbMin + (long)cell * PaddedDimensions;
        float* mx = bbMax + (long)cell * PaddedDimensions;
        var qv0 = Vector256.Load(q);
        var qv1 = Vector256.Load(q + 8);
        var mn0 = Vector256.Load(mn);
        var mn1 = Vector256.Load(mn + 8);
        var mx0 = Vector256.Load(mx);
        var mx1 = Vector256.Load(mx + 8);
        // dist[d] = max(0, max(mn-q, q-mx)); paddings (d>=14) have mn=mx=0 so dist=|q|, but
        // padded query dims are 0 too → contributes 0. Squared sum is then exact 14-dim LB.
        var d0a = mn0 - qv0; var d0b = qv0 - mx0;
        var d1a = mn1 - qv1; var d1b = qv1 - mx1;
        var zero = Vector256<float>.Zero;
        var d0 = Vector256.Max(zero, Vector256.Max(d0a, d0b));
        var d1 = Vector256.Max(zero, Vector256.Max(d1a, d1b));
        var s = (d0 * d0) + (d1 * d1);
        return Vector256.Sum(s);
    }
}
