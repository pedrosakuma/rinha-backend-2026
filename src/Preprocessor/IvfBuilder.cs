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
                "<ivf-centroids.bin> <ivf-offsets.bin> [nlist=256] [maxIters=20] [seed=42]");
            return 1;
        }

        var vectorsPath = args[0];
        var labelsPath = args[1];
        var q8Path = args[2];
        var centroidsPath = args[3];
        var offsetsPath = args[4];
        int nlist = args.Length > 5 ? int.Parse(args[5]) : 256;
        int maxIters = args.Length > 6 ? int.Parse(args[6]) : 20;
        int seed = args.Length > 7 ? int.Parse(args[7]) : 42;

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
