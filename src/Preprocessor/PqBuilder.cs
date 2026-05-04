using System.Diagnostics;
using System.Runtime.InteropServices;

namespace Rinha.Preprocessor;

/// <summary>
/// Builds Product Quantization (PQ) codebooks + codes over the (already reordered by IVF cell)
/// references.bin.
///
/// Layout: D=14 split into M sub-vectors of d_sub=D/M dims each (default M=7 → d_sub=2).
/// Each sub-vector is k-means quantized into ksub=256 codes (1 byte index).
///
/// Outputs:
///  - pq_codebooks.bin: M × ksub × d_sub × float (default 7×256×2×4 = 14 KB), contiguous (m, k, d).
///  - pq_codes.bin:     N × M bytes, row-major (one row per vector, M consecutive bytes).
///
/// The runtime scorer builds a per-query LUT[M][ksub] of squared distance from each sub-query to
/// each centroid in that subspace, then for each row sums M LUT lookups (Asymmetric Distance Computation,
/// the FAISS IVFPQ standard).
/// </summary>
public static unsafe class PqBuilder
{
    private const int Dimensions = 14;
    private const int PaddedDimensions = 16;

    public static int Run(string[] args)
    {
        if (args.Length < 3)
        {
            Console.Error.WriteLine(
                "Usage: Rinha.Preprocessor --pq <vectors.bin> <pq-codebooks.bin> <pq-codes.bin> " +
                "[M=7] [ksub=256] [maxIters=15] [seed=42]");
            return 1;
        }

        var vecPath = args[0];
        var cbPath = args[1];
        var codesPath = args[2];
        int M = args.Length > 3 ? int.Parse(args[3]) : 7;
        int ksub = args.Length > 4 ? int.Parse(args[4]) : 256;
        int maxIters = args.Length > 5 ? int.Parse(args[5]) : 15;
        int seed = args.Length > 6 ? int.Parse(args[6]) : 42;

        if (Dimensions % M != 0)
        {
            Console.Error.WriteLine($"M={M} must divide D={Dimensions}");
            return 2;
        }
        int dsub = Dimensions / M;

        var rowFloatBytes = PaddedDimensions * sizeof(float);
        var fi = new FileInfo(vecPath);
        if (fi.Length % rowFloatBytes != 0)
            throw new InvalidDataException($"Vectors size {fi.Length} not multiple of {rowFloatBytes}");
        long count = fi.Length / rowFloatBytes;
        if (count > int.MaxValue) throw new InvalidDataException("Too large");
        int n = (int)count;
        Console.Error.WriteLine($"PQ: N={n:N0} D={Dimensions} M={M} dsub={dsub} ksub={ksub}");

        var sw = Stopwatch.StartNew();
        var vectors = new float[(long)n * PaddedDimensions];
        ReadAll(vecPath, MemoryMarshal.AsBytes(vectors.AsSpan()));
        Console.Error.WriteLine($"  loaded vectors in {sw.Elapsed.TotalSeconds:F2}s");

        var codebooks = new float[M * ksub * dsub];
        var codes = new byte[(long)n * M];

        for (int m = 0; m < M; m++)
        {
            sw.Restart();
            int subOffset = m * dsub;
            var sub = TrainSubQuantizer(vectors, n, subOffset, dsub, ksub, maxIters, seed + m, out var assign);
            Array.Copy(sub, 0, codebooks, m * ksub * dsub, ksub * dsub);
            // codes[row*M + m] = assign[row]
            for (int i = 0; i < n; i++)
                codes[(long)i * M + m] = (byte)assign[i];
            Console.Error.WriteLine($"  m={m}: trained in {sw.Elapsed.TotalSeconds:F2}s");
        }

        sw.Restart();
        WriteAll(cbPath, MemoryMarshal.AsBytes(codebooks.AsSpan()));
        WriteAll(codesPath, codes);
        Console.Error.WriteLine($"  wrote in {sw.Elapsed.TotalSeconds:F2}s");
        Console.Error.WriteLine($"PQ done: codebooks={new FileInfo(cbPath).Length:N0}B codes={new FileInfo(codesPath).Length:N0}B");
        return 0;
    }

    /// <summary>
    /// k-means++ init + Lloyd on a d_sub-dimensional sub-vector extracted from each padded row at subOffset.
    /// Returns centroids[ksub * dsub] and per-row assignments via out param.
    /// </summary>
    private static float[] TrainSubQuantizer(float[] vectors, int n, int subOffset, int dsub, int ksub, int maxIters, int seed, out int[] assign)
    {
        var rng = new Random(seed);
        var centroids = new float[ksub * dsub];
        var localAssign = new int[n];
        assign = localAssign;

        // k-means++ init.
        int firstIdx = rng.Next(n);
        for (int d = 0; d < dsub; d++)
            centroids[d] = vectors[(long)firstIdx * PaddedDimensions + subOffset + d];

        var minDist = new float[n];
        Array.Fill(minDist, float.PositiveInfinity);
        UpdateMinDist(vectors, n, subOffset, dsub, centroids, 0, minDist);

        for (int k = 1; k < ksub; k++)
        {
            double total = 0;
            for (int i = 0; i < n; i++) total += minDist[i];
            int pick;
            if (total <= 0) pick = rng.Next(n);
            else
            {
                double r = rng.NextDouble() * total;
                double acc = 0;
                pick = n - 1;
                for (int i = 0; i < n; i++)
                {
                    acc += minDist[i];
                    if (acc >= r) { pick = i; break; }
                }
            }
            for (int d = 0; d < dsub; d++)
                centroids[k * dsub + d] = vectors[(long)pick * PaddedDimensions + subOffset + d];
            UpdateMinDist(vectors, n, subOffset, dsub, centroids, k, minDist);
        }

        // Lloyd iterations.
        var sums = new float[ksub * dsub];
        var counts = new int[ksub];
        for (int iter = 0; iter < maxIters; iter++)
        {
            int changed = 0;
            // Parallel assignment.
            Parallel.For<int>(
                0, n,
                () => 0,
                (i, _, local) =>
                {
                    long rowOff = (long)i * PaddedDimensions + subOffset;
                    int best = 0;
                    float bestDist = float.PositiveInfinity;
                    for (int k = 0; k < ksub; k++)
                    {
                        float d2 = 0;
                        int cOff = k * dsub;
                        for (int d = 0; d < dsub; d++)
                        {
                            float diff = vectors[rowOff + d] - centroids[cOff + d];
                            d2 += diff * diff;
                        }
                        if (d2 < bestDist) { bestDist = d2; best = k; }
                    }
                    if (localAssign[i] != best) { localAssign[i] = best; local++; }
                    return local;
                },
                local => Interlocked.Add(ref changed, local));

            Array.Clear(sums);
            Array.Clear(counts);
            for (int i = 0; i < n; i++)
            {
                int k = assign[i];
                counts[k]++;
                long rowOff = (long)i * PaddedDimensions + subOffset;
                int cOff = k * dsub;
                for (int d = 0; d < dsub; d++)
                    sums[cOff + d] += vectors[rowOff + d];
            }
            int empties = 0;
            for (int k = 0; k < ksub; k++)
            {
                int cOff = k * dsub;
                if (counts[k] > 0)
                {
                    float inv = 1f / counts[k];
                    for (int d = 0; d < dsub; d++)
                        centroids[cOff + d] = sums[cOff + d] * inv;
                }
                else
                {
                    empties++;
                    int pick = (int)((uint)((k + 1) * 2654435761u) % (uint)n);
                    for (int d = 0; d < dsub; d++)
                        centroids[cOff + d] = vectors[(long)pick * PaddedDimensions + subOffset + d];
                }
            }
            if (changed * 200 < n) break;
        }
        return centroids;
    }

    private static void UpdateMinDist(float[] vectors, int n, int subOffset, int dsub, float[] centroids, int k, float[] minDist)
    {
        int cOff = k * dsub;
        Parallel.For(0, n, i =>
        {
            long rowOff = (long)i * PaddedDimensions + subOffset;
            float d2 = 0;
            for (int d = 0; d < dsub; d++)
            {
                float diff = vectors[rowOff + d] - centroids[cOff + d];
                d2 += diff * diff;
            }
            if (d2 < minDist[i]) minDist[i] = d2;
        });
    }

    private static void ReadAll(string path, Span<byte> buffer)
    {
        using var fs = File.OpenRead(path);
        int total = 0;
        while (total < buffer.Length)
        {
            int read = fs.Read(buffer[total..]);
            if (read <= 0) throw new IOException($"Unexpected EOF: {path}");
            total += read;
        }
    }

    private static void WriteAll(string path, ReadOnlySpan<byte> buffer)
    {
        using var fs = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.None, 1 << 20, FileOptions.SequentialScan);
        fs.Write(buffer);
    }

    private static void WriteAll(string path, byte[] buffer) => WriteAll(path, (ReadOnlySpan<byte>)buffer.AsSpan());
}
