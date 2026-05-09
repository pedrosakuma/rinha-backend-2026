using System.Buffers.Binary;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;

namespace Rinha.Preprocessor;

/// <summary>
/// Builds an IVF (inverted-file) index over the already-converted reference dataset.
///
/// Inputs:  references.bin (N×16 float), labels.bin (N×byte), references_q8.bin (N×16 sbyte)
/// Outputs: rewrites the 3 input files reordered by cell (no extra space),
///          writes ivf_centroids.bin (NLIST×16 float) and
///          ivf_offsets.bin ((NLIST+1)×int32, prefix-sum so cell c covers [off[c], off[c+1])).
///
/// Algorithm: k-means++ init + Lloyd iterations (parallel assignment, sequential centroid update).
/// Distance: squared L2 over the 16-dim padded vector (padding zero in both q and centroid).
/// </summary>
public static unsafe class IvfBuilder
{
    private const int Dimensions = 14;
    private const int PaddedDimensions = 16;

    public static int Run(string[] args)
    {
        if (args.Length < 5)
        {
            Console.Error.WriteLine(
                "Usage: Rinha.Preprocessor --ivf <vectors.bin> <labels.bin> <vectors_q8.bin> " +
                "<ivf-centroids.bin> <ivf-offsets.bin> [<bbox-min.bin> <bbox-max.bin>] [nlist=256] [maxIters=20] [seed=42]");
            return 1;
        }

        var vectorsPath = args[0];
        var labelsPath = args[1];
        var q8Path = args[2];
        var centroidsPath = args[3];
        var offsetsPath = args[4];
        // Optional bbox paths (J6: bbox lower-bound exact repair).
        // If provided, must come BEFORE positional numeric args. Detect by ".bin" suffix.
        string? bboxMinPath = null, bboxMaxPath = null;
        string? q8SoaPath = null; // J11: SoA layout for scalar early-abort
        string? q16Path = null;   // J25: int16 rerank (replaces float on hot path)
        string? q16SoaPath = null; // column-major Q16 for brute-force AVX2 scan (pre-transposed)
        int positional = 5;
        if (args.Length > 5 && args[5].EndsWith(".bin", StringComparison.OrdinalIgnoreCase)
            && args.Length > 6 && args[6].EndsWith(".bin", StringComparison.OrdinalIgnoreCase))
        {
            bboxMinPath = args[5];
            bboxMaxPath = args[6];
            positional = 7;
            if (args.Length > 7 && args[7].EndsWith(".bin", StringComparison.OrdinalIgnoreCase))
            {
                q8SoaPath = args[7];
                positional = 8;
                if (args.Length > 8 && args[8].EndsWith(".bin", StringComparison.OrdinalIgnoreCase))
                {
                    q16Path = args[8];
                    positional = 9;
                    if (args.Length > 9 && args[9].EndsWith(".bin", StringComparison.OrdinalIgnoreCase))
                    {
                        q16SoaPath = args[9];
                        positional = 10;
                    }
                }
            }
        }
        int nlist = args.Length > positional ? int.Parse(args[positional]) : 256;
        int maxIters = args.Length > positional + 1 ? int.Parse(args[positional + 1]) : 20;
        int seed = args.Length > positional + 2 ? int.Parse(args[positional + 2]) : 42;
        // Cell-size cap = ceil(N/nlist * (1 + balanceSlack)). 0 disables (default).
        double balanceSlack = args.Length > positional + 3 ? double.Parse(args[positional + 3], System.Globalization.CultureInfo.InvariantCulture) : 0.0;

        var rowFloatBytes = PaddedDimensions * sizeof(float);
        var rowQ8Bytes = PaddedDimensions;

        var vecsLen = new FileInfo(vectorsPath).Length;
        var labelsLen = new FileInfo(labelsPath).Length;
        var q8Len = new FileInfo(q8Path).Length;
        if (vecsLen % rowFloatBytes != 0)
            throw new InvalidDataException($"Vectors size {vecsLen} not multiple of {rowFloatBytes}");
        long count = vecsLen / rowFloatBytes;
        if (labelsLen != count) throw new InvalidDataException($"Labels {labelsLen} != count {count}");
        if (q8Len != count * rowQ8Bytes) throw new InvalidDataException($"Q8 {q8Len} != expected");
        if (count > int.MaxValue) throw new InvalidDataException("Too large");

        int n = (int)count;
        Console.Error.WriteLine($"IVF: N={n:N0} D={Dimensions} (padded {PaddedDimensions}) NLIST={nlist}");

        var sw = Stopwatch.StartNew();
        var vectors = new float[(long)n * PaddedDimensions];
        var labels = new byte[n];
        var q8 = new sbyte[(long)n * PaddedDimensions];
        ReadAll(vectorsPath, MemoryMarshal.AsBytes(vectors.AsSpan()));
        ReadAll(labelsPath, labels);
        ReadAll(q8Path, MemoryMarshal.AsBytes(q8.AsSpan()));
        Console.Error.WriteLine($"  loaded in {sw.Elapsed.TotalSeconds:F2}s");

        sw.Restart();
        var centroids = KMeansPlusPlusInit(vectors, n, nlist, seed);
        Console.Error.WriteLine($"  init in {sw.Elapsed.TotalSeconds:F2}s");

        var assign = new int[n];
        sw.Restart();
        Lloyd(vectors, n, centroids, assign, nlist, maxIters);
        Console.Error.WriteLine($"  Lloyd in {sw.Elapsed.TotalSeconds:F2}s");

        // J18: post-hoc split of heavy cells. Preserves natural clusters but subdivides
        // any cell exceeding IVF_HEAVY_SPLIT_MAX rows via local k-means. New centroids
        // are appended; nlist grows.
        int heavyMax = int.TryParse(Environment.GetEnvironmentVariable("IVF_HEAVY_SPLIT_MAX") ?? "0", out var hm) ? hm : 0;
        if (heavyMax > 0)
        {
            sw.Restart();
            int splits = HeavySplit(vectors, n, ref centroids, assign, ref nlist, heavyMax, seed);
            Console.Error.WriteLine($"  heavy-split (max={heavyMax}) in {sw.Elapsed.TotalSeconds:F2}s, splits={splits}, new nlist={nlist}");
        }

        if (balanceSlack > 0)
        {
            sw.Restart();
            int cap = (int)Math.Ceiling((double)n / nlist * (1.0 + balanceSlack));
            int moved = Rebalance(vectors, n, centroids, assign, nlist, cap);
            Console.Error.WriteLine($"  rebalance (cap={cap}, slack={balanceSlack:F2}) in {sw.Elapsed.TotalSeconds:F2}s, moved={moved:N0}");
            // Recompute centroids as means of post-rebalance cells, then run a few more Lloyd
            // iters with cap-enforcement on assignment (penalize over-cap cells).
            sw.Restart();
            RecomputeCentroidsFromAssign(vectors, n, centroids, assign, nlist);
            Console.Error.WriteLine($"  recenter post-rebalance in {sw.Elapsed.TotalSeconds:F2}s");
        }

        sw.Restart();
        var (newVecs, newLabs, newQ8, offsets) = Reorder(vectors, labels, q8, assign, nlist);
        Console.Error.WriteLine($"  reorder in {sw.Elapsed.TotalSeconds:F2}s");

        sw.Restart();
        WriteAll(vectorsPath, MemoryMarshal.AsBytes(newVecs.AsSpan()));
        WriteAll(labelsPath, newLabs);
        WriteAll(q8Path, MemoryMarshal.AsBytes(newQ8.AsSpan()));
        WriteAll(centroidsPath, MemoryMarshal.AsBytes(centroids.AsSpan()));
        WriteOffsets(offsetsPath, offsets);
        Console.Error.WriteLine($"  wrote in {sw.Elapsed.TotalSeconds:F2}s");

        // J6: per-cluster bbox (min/max per dim) for exact lower-bound repair.
        if (bboxMinPath is not null && bboxMaxPath is not null)
        {
            sw.Restart();
            var (bboxMin, bboxMax) = ComputeBboxes(newVecs, offsets, nlist);
            WriteAll(bboxMinPath, MemoryMarshal.AsBytes(bboxMin.AsSpan()));
            WriteAll(bboxMaxPath, MemoryMarshal.AsBytes(bboxMax.AsSpan()));
            Console.Error.WriteLine($"  bbox in {sw.Elapsed.TotalSeconds:F2}s ({nlist}x{PaddedDimensions} floats each)");
        }

        // J11: SoA layout for scalar early-abort. 14 contiguous blocks of N bytes,
        // block d = dim d's q8 value for row 0..N-1 (rows already in IVF order).
        // Cache-friendly for per-dim sequential streaming with row-survivor compaction.
        if (q8SoaPath is not null)
        {
            sw.Restart();
            int total = n * Dimensions;
            var soa = new sbyte[total];
            for (int d = 0; d < Dimensions; d++)
            {
                int dimBase = d * n;
                for (int r = 0; r < n; r++)
                {
                    soa[dimBase + r] = newQ8[(long)r * PaddedDimensions + d];
                }
            }
            WriteAll(q8SoaPath, MemoryMarshal.AsBytes(soa.AsSpan()));
            Console.Error.WriteLine($"  q8-soa in {sw.Elapsed.TotalSeconds:F2}s ({Dimensions}x{n} bytes)");
        }

        // J25: Q16 (int16) layout for high-precision rerank (replaces float on hot path).
        // Scale = 10000 (queries are pre-rounded to 4dp, references upstream are also 4dp).
        // Range used: [-10000, 10000] fits comfortably in int16. Sentinel -1f → -10000.
        // Padding (dims 14..15) → 0. Storage: N * 16 * 2 = 32 bytes/row (96MB total),
        // vs 64 bytes/row for float. Net working-set save: ~96MB.
        if (q16Path is not null)
        {
            sw.Restart();
            var q16 = new short[(long)n * PaddedDimensions];
            for (int r = 0; r < n; r++)
            {
                long baseSrc = (long)r * PaddedDimensions;
                long baseDst = (long)r * PaddedDimensions;
                for (int d = 0; d < Dimensions; d++)
                {
                    float v = newVecs[baseSrc + d];
                    int q = (int)MathF.Round(v * 10000f);
                    if (q > 32767) q = 32767;
                    else if (q < -32768) q = -32768;
                    q16[baseDst + d] = (short)q;
                }
                // padding lanes already zero (default-initialized array)
            }
            WriteAll(q16Path, MemoryMarshal.AsBytes(q16.AsSpan()));
            Console.Error.WriteLine($"  q16 in {sw.Elapsed.TotalSeconds:F2}s ({n}x{PaddedDimensions} shorts = {(long)n * PaddedDimensions * 2:N0} bytes)");

            // Column-major (SoA) Q16 for brute-force AVX2 scan.
            // Layout: q16Soa[d * n + r] = Q16 value of vector r at dimension d.
            // Size: Dimensions * n * sizeof(short) = 14 * n * 2 (no padding cols).
            if (q16SoaPath is not null)
            {
                sw.Restart();
                var q16Soa = new short[(long)Dimensions * n];
                for (int d = 0; d < Dimensions; d++)
                {
                    long colBase = (long)d * n;
                    for (int r = 0; r < n; r++)
                        q16Soa[colBase + r] = q16[(long)r * PaddedDimensions + d];
                }
                WriteAll(q16SoaPath, MemoryMarshal.AsBytes(q16Soa.AsSpan()));
                Console.Error.WriteLine($"  q16-soa in {sw.Elapsed.TotalSeconds:F2}s ({Dimensions}x{n} shorts = {(long)Dimensions * n * 2:N0} bytes)");
            }
        }

        // Stats
        int empty = 0, maxCell = 0;
        for (int c = 0; c < nlist; c++)
        {
            int sz = offsets[c + 1] - offsets[c];
            if (sz == 0) empty++;
            if (sz > maxCell) maxCell = sz;
        }
        double mean = (double)n / nlist;
        Console.Error.WriteLine($"IVF done: nlist={nlist} mean={mean:F0} max={maxCell} empty={empty}");
        return 0;
    }

    private static float[] KMeansPlusPlusInit(float[] vectors, int n, int nlist, int seed)
    {
        var rng = new Random(seed);
        var centroids = new float[(long)nlist * PaddedDimensions];

        // Pick first centroid uniformly.
        int first = rng.Next(n);
        Array.Copy(vectors, (long)first * PaddedDimensions, centroids, 0, PaddedDimensions);

        // Track squared dist to nearest existing centroid.
        var minDist = new float[n];
        Array.Fill(minDist, float.PositiveInfinity);

        fixed (float* vBase = vectors)
        fixed (float* cBase = centroids)
        {
            UpdateMinDist(vBase, n, cBase + 0, minDist);

            for (int k = 1; k < nlist; k++)
            {
                // Sample next center proportional to minDist.
                double total = 0;
                for (int i = 0; i < n; i++) total += minDist[i];
                if (total <= 0)
                {
                    // Degenerate: pick random.
                    int pick = rng.Next(n);
                    Array.Copy(vectors, (long)pick * PaddedDimensions, centroids, (long)k * PaddedDimensions, PaddedDimensions);
                }
                else
                {
                    double r = rng.NextDouble() * total;
                    double acc = 0;
                    int pick = n - 1;
                    for (int i = 0; i < n; i++)
                    {
                        acc += minDist[i];
                        if (acc >= r) { pick = i; break; }
                    }
                    Array.Copy(vectors, (long)pick * PaddedDimensions, centroids, (long)k * PaddedDimensions, PaddedDimensions);
                }
                UpdateMinDist(vBase, n, cBase + (long)k * PaddedDimensions, minDist);
            }
        }
        return centroids;
    }

    private static void UpdateMinDist(float* vBase, int n, float* centroid, float[] minDist)
    {
        var c0 = Vector256.Load(centroid);
        var c1 = Vector256.Load(centroid + 8);
        // Parallelize the scan; minDist updates per-i are independent.
        Parallel.For(0, n, i =>
        {
            float* row = vBase + (long)i * PaddedDimensions;
            var r0 = Vector256.Load(row);
            var r1 = Vector256.Load(row + 8);
            var d0 = r0 - c0;
            var d1 = r1 - c1;
            var s = (d0 * d0) + (d1 * d1);
            float dist = Vector256.Sum(s);
            if (dist < minDist[i]) minDist[i] = dist;
        });
    }

    private static void Lloyd(float[] vectors, int n, float[] centroids, int[] assign, int nlist, int maxIters)
    {
        var sums = new float[(long)nlist * PaddedDimensions];
        var counts = new int[nlist];

        // Pin once so parallel workers can use raw pointers (fixed-locals can't be captured).
        var vHandle = GCHandle.Alloc(vectors, GCHandleType.Pinned);
        var cHandle = GCHandle.Alloc(centroids, GCHandleType.Pinned);
        try
        {
            float* vBase = (float*)vHandle.AddrOfPinnedObject();
            float* cBase = (float*)cHandle.AddrOfPinnedObject();

            for (int iter = 0; iter < maxIters; iter++)
            {
                int changed = 0;
                Parallel.For<int>(
                    0, n,
                    () => 0,
                    (i, _, local) =>
                    {
                        float* row = vBase + (long)i * PaddedDimensions;
                        var r0 = Vector256.Load(row);
                        var r1 = Vector256.Load(row + 8);
                        int best = 0;
                        float bestDist = float.PositiveInfinity;
                        for (int c = 0; c < nlist; c++)
                        {
                            float* cp = cBase + (long)c * PaddedDimensions;
                            var k0 = Vector256.Load(cp);
                            var k1 = Vector256.Load(cp + 8);
                            var d0 = r0 - k0;
                            var d1 = r1 - k1;
                            var s = (d0 * d0) + (d1 * d1);
                            float d = Vector256.Sum(s);
                            if (d < bestDist) { bestDist = d; best = c; }
                        }
                        if (assign[i] != best) { assign[i] = best; local++; }
                        return local;
                    },
                    local => Interlocked.Add(ref changed, local));

                Array.Clear(sums);
                Array.Clear(counts);
                for (int i = 0; i < n; i++)
                {
                    int c = assign[i];
                    counts[c]++;
                    long src = (long)i * PaddedDimensions;
                    long dst = (long)c * PaddedDimensions;
                    for (int d = 0; d < Dimensions; d++)
                        sums[dst + d] += vectors[src + d];
                }
                int empties = 0;
                for (int c = 0; c < nlist; c++)
                {
                    long dst = (long)c * PaddedDimensions;
                    if (counts[c] > 0)
                    {
                        float inv = 1f / counts[c];
                        for (int d = 0; d < Dimensions; d++)
                            centroids[dst + d] = sums[dst + d] * inv;
                    }
                    else
                    {
                        empties++;
                        int pick = (int)((uint)(c * 2654435761u) % (uint)n);
                        Array.Copy(vectors, (long)pick * PaddedDimensions, centroids, dst, PaddedDimensions);
                    }
                }

                Console.Error.WriteLine($"  iter {iter}: changed={changed:N0} ({100.0 * changed / n:F2}%) empties={empties}");
                if (changed * 200 < n) break;
            }
        }
        finally
        {
            vHandle.Free();
            cHandle.Free();
        }
    }

    /// <summary>
    /// Cap-based balancing. For each over-cap cell, compute distance from each member to its
    /// centroid, keep the closest <c>cap</c> members, and reassign the kicked tail to the nearest
    /// centroid with remaining slack (full nlist scan, since precomputed top-K may all be full).
    /// Iterates until no over-cap remains or no progress is possible.
    /// </summary>
    private static int Rebalance(float[] vectors, int n, float[] centroids, int[] assign, int nlist, int cap)
    {
        var size = new int[nlist];
        for (int i = 0; i < n; i++) size[assign[i]]++;

        var vHandle = GCHandle.Alloc(vectors, GCHandleType.Pinned);
        var cHandle = GCHandle.Alloc(centroids, GCHandleType.Pinned);
        int totalMoved = 0;
        try
        {
            float* vPtr = (float*)vHandle.AddrOfPinnedObject();
            float* cPtr = (float*)cHandle.AddrOfPinnedObject();

            for (int pass = 0; pass < 30; pass++)
            {
                int over = 0;
                for (int c = 0; c < nlist; c++) if (size[c] > cap) over++;
                if (over == 0) break;

                int movedThisPass = 0;
                int curMaxBefore = 0;
                for (int c = 0; c < nlist; c++) if (size[c] > curMaxBefore) curMaxBefore = size[c];

                for (int c = 0; c < nlist; c++)
                {
                    if (size[c] <= cap) continue;
                    int excess = size[c] - cap;

                    // Gather all members of cell c with their dist to c.
                    var members = new (int idx, float dist)[size[c]];
                    int mi = 0;
                    float* cp = cPtr + (long)c * PaddedDimensions;
                    var k0 = Vector256.Load(cp);
                    var k1 = Vector256.Load(cp + 8);
                    for (int i = 0; i < n; i++)
                    {
                        if (assign[i] != c) continue;
                        float* row = vPtr + (long)i * PaddedDimensions;
                        var r0 = Vector256.Load(row);
                        var r1 = Vector256.Load(row + 8);
                        var d0 = k0 - r0;
                        var d1 = k1 - r1;
                        var s = (d0 * d0) + (d1 * d1);
                        members[mi++] = (i, Vector256.Sum(s));
                    }
                    // Sort ascending by dist; keep first cap, kick the rest.
                    Array.Sort(members, (a, b) => a.dist.CompareTo(b.dist));

                    // Walk kicked tail (furthest) and find nearest centroid with slack.
                    for (int t = members.Length - 1; t >= cap && excess > 0; t--)
                    {
                        int i = members[t].idx;
                        float* row = vPtr + (long)i * PaddedDimensions;
                        var r0 = Vector256.Load(row);
                        var r1 = Vector256.Load(row + 8);
                        int bestC = -1;
                        float bestD = float.PositiveInfinity;
                        for (int alt = 0; alt < nlist; alt++)
                        {
                            if (alt == c) continue;
                            if (size[alt] >= cap) continue;
                            float* ap = cPtr + (long)alt * PaddedDimensions;
                            var a0 = Vector256.Load(ap);
                            var a1 = Vector256.Load(ap + 8);
                            var dd0 = a0 - r0;
                            var dd1 = a1 - r1;
                            var ss = (dd0 * dd0) + (dd1 * dd1);
                            float dist = Vector256.Sum(ss);
                            if (dist < bestD) { bestD = dist; bestC = alt; }
                        }
                        if (bestC < 0) break;
                        assign[i] = bestC;
                        size[c]--;
                        size[bestC]++;
                        excess--;
                        movedThisPass++;
                    }
                }

                int curMax = 0;
                for (int c = 0; c < nlist; c++) if (size[c] > curMax) curMax = size[c];
                Console.Error.WriteLine($"    pass {pass + 1}: over={over} moved={movedThisPass:N0} max={curMaxBefore}->{curMax}");
                totalMoved += movedThisPass;
                if (movedThisPass == 0) break;
            }
        }
        finally
        {
            cHandle.Free(); vHandle.Free();
        }
        return totalMoved;
    }

    /// <summary>
    /// Re-set each centroid to the mean of its current assigned points. Used after rebalancing
    /// so the centroids reflect the post-balance cell composition (otherwise they still point to
    /// the original Lloyd clusters and IVF cell selection misses force-moved points at query time).
    /// </summary>
    private static void RecomputeCentroidsFromAssign(float[] vectors, int n, float[] centroids, int[] assign, int nlist)
    {
        var sums = new double[(long)nlist * PaddedDimensions];
        var counts = new int[nlist];
        for (int i = 0; i < n; i++)
        {
            int c = assign[i];
            counts[c]++;
            long src = (long)i * PaddedDimensions;
            long dst = (long)c * PaddedDimensions;
            for (int d = 0; d < PaddedDimensions; d++) sums[dst + d] += vectors[src + d];
        }
        for (int c = 0; c < nlist; c++)
        {
            if (counts[c] == 0) continue;
            long dst = (long)c * PaddedDimensions;
            float inv = 1.0f / counts[c];
            for (int d = 0; d < PaddedDimensions; d++) centroids[dst + d] = (float)(sums[dst + d] * inv);
        }
    }

    /// <summary>
    /// Post-hoc split of heavy cells: any cell with size &gt; maxSize is partitioned
    /// into ceil(size/maxSize) sub-cells via local k-means. Preserves natural clusters
    /// (only oversized cells are touched). New centroids are appended; nlist grows.
    /// Returns number of cells split.
    /// </summary>
    private static int HeavySplit(float[] vectors, int n, ref float[] centroids,
        int[] assign, ref int nlist, int maxSize, int seed)
    {
        // Build per-cell index lists (snapshot of current state).
        var perCell = new List<int>[nlist];
        for (int i = 0; i < nlist; i++) perCell[i] = new List<int>();
        for (int i = 0; i < n; i++) perCell[assign[i]].Add(i);

        var heavy = new List<int>();
        int extra = 0;
        for (int c = 0; c < nlist; c++)
        {
            int sz = perCell[c].Count;
            if (sz > maxSize)
            {
                heavy.Add(c);
                int parts = (sz + maxSize - 1) / maxSize;
                extra += parts - 1;
            }
        }
        if (heavy.Count == 0) return 0;

        int newNlist = nlist + extra;
        Array.Resize(ref centroids, newNlist * PaddedDimensions);

        var rng = new Random(seed * 7 + 13);
        int nextIdx = nlist;
        const int LocalIters = 12;
        var sub = new float[16 * PaddedDimensions]; // worst case: ~14 sub-clusters per cell
        var subAssign = new int[1];                 // resized per cell
        var sums = new float[16 * PaddedDimensions];
        var counts = new int[16];

        foreach (var c in heavy)
        {
            var members = perCell[c];
            int m = members.Count;
            int splitN = (m + maxSize - 1) / maxSize;
            if (splitN > 16) splitN = 16; // hard cap; very unlikely

            // Lazy resize buffers if needed (we sized for 16; fine).
            if (subAssign.Length < m) subAssign = new int[m];

            // Init sub-centroids by picking splitN distinct random members.
            var picks = new HashSet<int>();
            for (int s = 0; s < splitN; s++)
            {
                int p; do { p = rng.Next(m); } while (!picks.Add(p));
                int row = members[p];
                Array.Copy(vectors, (long)row * PaddedDimensions,
                    sub, s * PaddedDimensions, PaddedDimensions);
            }

            for (int it = 0; it < LocalIters; it++)
            {
                // Assignment.
                for (int mi = 0; mi < m; mi++)
                {
                    long rowOff = (long)members[mi] * PaddedDimensions;
                    float bestD = float.PositiveInfinity; int bestS = 0;
                    for (int s = 0; s < splitN; s++)
                    {
                        int sOff = s * PaddedDimensions;
                        float d = 0;
                        for (int dim = 0; dim < Dimensions; dim++)
                        {
                            float diff = vectors[rowOff + dim] - sub[sOff + dim];
                            d += diff * diff;
                        }
                        if (d < bestD) { bestD = d; bestS = s; }
                    }
                    subAssign[mi] = bestS;
                }
                // Recompute means.
                Array.Clear(sums, 0, splitN * PaddedDimensions);
                Array.Clear(counts, 0, splitN);
                for (int mi = 0; mi < m; mi++)
                {
                    int s = subAssign[mi];
                    counts[s]++;
                    long rowOff = (long)members[mi] * PaddedDimensions;
                    int sOff = s * PaddedDimensions;
                    for (int dim = 0; dim < Dimensions; dim++)
                        sums[sOff + dim] += vectors[rowOff + dim];
                }
                for (int s = 0; s < splitN; s++)
                {
                    if (counts[s] == 0) continue;
                    float inv = 1.0f / counts[s];
                    int sOff = s * PaddedDimensions;
                    for (int dim = 0; dim < Dimensions; dim++)
                        sub[sOff + dim] = sums[sOff + dim] * inv;
                }
            }

            // Apply: sub 0 inherits original cell id `c`, others get fresh ids.
            var newIds = new int[splitN];
            newIds[0] = c;
            Array.Copy(sub, 0, centroids, (long)c * PaddedDimensions, PaddedDimensions);
            for (int s = 1; s < splitN; s++)
            {
                newIds[s] = nextIdx;
                Array.Copy(sub, s * PaddedDimensions,
                    centroids, (long)nextIdx * PaddedDimensions, PaddedDimensions);
                nextIdx++;
            }
            for (int mi = 0; mi < m; mi++)
                assign[members[mi]] = newIds[subAssign[mi]];
        }

        nlist = newNlist;
        return heavy.Count;
    }

    private static (float[] vecs, byte[] labs, sbyte[] q8s, int[] offsets) Reorder(
        float[] vectors, byte[] labels, sbyte[] q8, int[] assign, int nlist)
    {
        int n = labels.Length;
        var counts = new int[nlist];
        for (int i = 0; i < n; i++) counts[assign[i]]++;

        var offsets = new int[nlist + 1];
        for (int c = 0; c < nlist; c++) offsets[c + 1] = offsets[c] + counts[c];

        var newVecs = new float[(long)n * PaddedDimensions];
        var newLabs = new byte[n];
        var newQ8 = new sbyte[(long)n * PaddedDimensions];
        var cursor = (int[])offsets.Clone(); // running write position per cell

        for (int i = 0; i < n; i++)
        {
            int c = assign[i];
            int dst = cursor[c]++;
            Array.Copy(vectors, (long)i * PaddedDimensions, newVecs, (long)dst * PaddedDimensions, PaddedDimensions);
            Array.Copy(q8, (long)i * PaddedDimensions, newQ8, (long)dst * PaddedDimensions, PaddedDimensions);
            newLabs[dst] = labels[i];
        }

        return (newVecs, newLabs, newQ8, offsets);
    }

    private static (float[] bboxMin, float[] bboxMax) ComputeBboxes(float[] vecs, int[] offsets, int nlist)
    {
        long stride = PaddedDimensions;
        var bMin = new float[(long)nlist * PaddedDimensions];
        var bMax = new float[(long)nlist * PaddedDimensions];
        // Initialize each cluster's bbox: min=+inf, max=-inf for active dims; padded dims = 0.
        for (int c = 0; c < nlist; c++)
        {
            for (int j = 0; j < Dimensions; j++)
            {
                bMin[(long)c * stride + j] = float.PositiveInfinity;
                bMax[(long)c * stride + j] = float.NegativeInfinity;
            }
        }
        for (int c = 0; c < nlist; c++)
        {
            int s = offsets[c], e = offsets[c + 1];
            if (e <= s)
            {
                // Empty cluster: leave +inf/-inf so any LB query returns +inf → never scanned.
                continue;
            }
            for (int i = s; i < e; i++)
            {
                long row = (long)i * stride;
                for (int j = 0; j < Dimensions; j++)
                {
                    float v = vecs[row + j];
                    long bi = (long)c * stride + j;
                    if (v < bMin[bi]) bMin[bi] = v;
                    if (v > bMax[bi]) bMax[bi] = v;
                }
            }
        }
        return (bMin, bMax);
    }

    private static void ReadAll(string path, Span<byte> buffer)
    {
        using var fs = File.OpenRead(path);
        int total = 0;
        while (total < buffer.Length)
        {
            int read = fs.Read(buffer[total..]);
            if (read <= 0) throw new EndOfFileException(path);
            total += read;
        }
    }

    private static void ReadAll(string path, byte[] buffer) => ReadAll(path, buffer.AsSpan());

    private static void WriteAll(string path, ReadOnlySpan<byte> buffer)
    {
        using var fs = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.None, 1 << 20, FileOptions.SequentialScan);
        fs.Write(buffer);
    }

    private static void WriteAll(string path, byte[] buffer) => WriteAll(path, (ReadOnlySpan<byte>)buffer.AsSpan());

    private static void WriteOffsets(string path, int[] offsets)
    {
        var buf = new byte[offsets.Length * sizeof(int)];
        for (int i = 0; i < offsets.Length; i++)
            BinaryPrimitives.WriteInt32LittleEndian(buf.AsSpan(i * 4, 4), offsets[i]);
        WriteAll(path, buf);
    }

    private sealed class EndOfFileException : IOException
    {
        public EndOfFileException(string path) : base($"Unexpected end of file: {path}") { }
    }
}
