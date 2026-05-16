using System.Collections.Concurrent;
using System.Diagnostics;
using System.Globalization;
using System.Runtime.Intrinsics;
using System.Text;
using System.Text.Json;
using Rinha.Api;

namespace Rinha.Bench;

/// <summary>
/// Precision PoC: compares fraud-count agreement between baseline (float32 refs+query)
/// and FP16-quantized (refs and query passed through (float)(Half)v) using
/// brute-force top-5 scoring. Goal: decide if storing references as FP16 (with
/// F16C vcvtph2ps conversion) is precision-safe for the eval (0 FP/FN target).
///
/// Usage:
///   Rinha.Bench --fp16-poc --test-data=/tmp/rinha-eval/test/test-data.json \
///       [--sample=5000] [--seed=42] [--data-dir=./data]
/// </summary>
public static class Fp16Poc
{
    private const int K = 5;
    private const int PaddedDimensions = 16;

    public static int Run(string[] args)
    {
        string testData = "/tmp/rinha-eval/test/test-data.json";
        string dataDir = Path.Combine(FindRepoRoot(), "data");
        int sample = 5000;
        int seed = 42;
        foreach (var a in args)
        {
            if (a.StartsWith("--test-data=")) testData = a[12..];
            else if (a.StartsWith("--data-dir=")) dataDir = a[11..];
            else if (a.StartsWith("--sample=")) sample = int.Parse(a[9..]);
            else if (a.StartsWith("--seed=")) seed = int.Parse(a[7..]);
        }

        var vec = Path.Combine(dataDir, "references.bin");
        var lab = Path.Combine(dataDir, "labels.bin");
        if (!File.Exists(vec) || !File.Exists(lab))
        {
            Console.Error.WriteLine($"Missing references.bin/labels.bin in {dataDir}");
            return 2;
        }

        using var dataset = Dataset.Open(vec, lab);

        var root = FindRepoRoot();
        var norm = NormalizationConstants.Load(Path.Combine(root, "resources/normalization.json"));
        var mcc = MccRiskTable.Load(Path.Combine(root, "resources/mcc_risk.json"));
        var jvec = new JsonVectorizer(norm, mcc);

        Console.Error.WriteLine($"Loading queries from {testData} ...");
        var queries = LoadQueries(testData, jvec);
        Console.Error.WriteLine($"Loaded {queries.Count} queries.");

        // Sub-sample.
        if (sample > 0 && sample < queries.Count)
        {
            var rng = new Random(seed);
            queries = queries.OrderBy(_ => rng.Next()).Take(sample).ToList();
            Console.Error.WriteLine($"Sub-sampled {queries.Count} queries (seed={seed}).");
        }

        int count = dataset.Count;
        Console.Error.WriteLine($"Refs: {count}, allocating FP16-quantized parallel array ({(long)count * PaddedDimensions * 4 / (1024 * 1024)} MB)...");

        // Pre-compute FP16-quantized refs: each float passed through (float)(Half)v.
        var quantRefs = new float[(long)count * PaddedDimensions];
        unsafe
        {
            float* src = dataset.VectorsPtr;
            nint srcAddr = (nint)src;
            Parallel.For(0, count, i =>
            {
                float* s = (float*)srcAddr + (long)i * PaddedDimensions;
                int baseIdx = i * PaddedDimensions;
                for (int j = 0; j < Dataset.Dimensions; j++)
                    quantRefs[baseIdx + j] = (float)(Half)s[j];
                quantRefs[baseIdx + 14] = 0f;
                quantRefs[baseIdx + 15] = 0f;
            });
        }
        Console.Error.WriteLine("Quantization done. Running comparison...");

        int totalDiffCount = 0;          // queries where fraud_count differs
        int totalTopKDiff = 0;            // queries where top-5 idx set differs
        var countDeltaHist = new ConcurrentDictionary<int, int>(); // delta in fraud count: −5..+5
        var stopwatch = Stopwatch.StartNew();
        int processed = 0;

        unsafe
        {
            float* baseRefs = dataset.VectorsPtr;
            byte* labels = dataset.LabelsPtr;
            fixed (float* quantRefsPin = quantRefs)
            {
                float* qRefs = quantRefsPin;

                Parallel.ForEach(queries, () => (diff: 0, topkDiff: 0, hist: new Dictionary<int, int>()),
                    (qr, _, local) =>
                    {
                        // Baseline: float query + float refs.
                        Span<float> qFloat = stackalloc float[PaddedDimensions];
                        for (int i = 0; i < Dataset.Dimensions; i++) qFloat[i] = qr.Vec[i];

                        // Quantized: query passed through (float)(Half)v.
                        Span<float> qFp16 = stackalloc float[PaddedDimensions];
                        for (int i = 0; i < Dataset.Dimensions; i++) qFp16[i] = (float)(Half)qr.Vec[i];

                        Span<int> topA = stackalloc int[K];
                        Span<int> topB = stackalloc int[K];
                        fixed (float* qaPtr = qFloat) fixed (float* qbPtr = qFp16)
                        {
                            BruteTopK(baseRefs, count, qaPtr, topA);
                            BruteTopK(qRefs, count, qbPtr, topB);
                        }
                        int fa = CountFrauds(topA, labels);
                        int fb = CountFrauds(topB, labels);
                        int delta = fb - fa;
                        if (!local.hist.TryAdd(delta, 1)) local.hist[delta]++;
                        int diff = fa != fb ? 1 : 0;
                        int topkDiff = TopKDiffers(topA, topB) ? 1 : 0;
                        return (local.diff + diff, local.topkDiff + topkDiff, local.hist);
                    },
                    local =>
                    {
                        Interlocked.Add(ref totalDiffCount, local.diff);
                        Interlocked.Add(ref totalTopKDiff, local.topkDiff);
                        foreach (var kv in local.hist)
                            countDeltaHist.AddOrUpdate(kv.Key, kv.Value, (_, v) => v + kv.Value);
                        Interlocked.Add(ref processed, local.diff + local.topkDiff); // dummy
                    });
            }
        }

        stopwatch.Stop();
        Console.WriteLine();
        Console.WriteLine($"=== FP16 PoC results ({queries.Count} queries) ===");
        Console.WriteLine($"Top-5 set divergence : {totalTopKDiff,6} ({(100.0 * totalTopKDiff / queries.Count):F4}%)");
        Console.WriteLine($"Fraud-count diverge  : {totalDiffCount,6} ({(100.0 * totalDiffCount / queries.Count):F4}%)");
        Console.WriteLine($"Elapsed              : {stopwatch.Elapsed.TotalSeconds:F1}s");
        Console.WriteLine();
        Console.WriteLine("Fraud-count delta histogram (fp16 - baseline):");
        foreach (var kv in countDeltaHist.OrderBy(k => k.Key))
            Console.WriteLine($"  delta={kv.Key,3}: {kv.Value,6} ({(100.0 * kv.Value / queries.Count):F4}%)");

        return 0;
    }

    private static unsafe void BruteTopK(float* refs_, int count, float* qPtr, Span<int> topIdx)
    {
        Span<float> topDist = stackalloc float[K];
        for (int i = 0; i < K; i++) { topDist[i] = float.PositiveInfinity; topIdx[i] = -1; }

        var q0 = Vector256.Load(qPtr);
        var q1 = Vector256.Load(qPtr + 8);
        float worst = float.PositiveInfinity;

        for (int i = 0; i < count; i++)
        {
            float* row = refs_ + (long)i * PaddedDimensions;
            var r0 = Vector256.Load(row);
            var r1 = Vector256.Load(row + 8);
            var d0 = r0 - q0;
            var d1 = r1 - q1;
            var sum = (d0 * d0) + (d1 * d1);
            float dist = Vector256.Sum(sum);
            if (dist < worst)
            {
                InsertTopK(topDist, topIdx, dist, i);
                worst = topDist[K - 1];
            }
        }
    }

    private static void InsertTopK(Span<float> dist, Span<int> idx, float newDist, int newIdx)
    {
        int pos = K - 1;
        while (pos > 0 && dist[pos - 1] > newDist) pos--;
        for (int j = K - 1; j > pos; j--) { dist[j] = dist[j - 1]; idx[j] = idx[j - 1]; }
        dist[pos] = newDist;
        idx[pos] = newIdx;
    }

    private static unsafe int CountFrauds(ReadOnlySpan<int> idx, byte* labels)
    {
        int n = 0;
        for (int i = 0; i < K; i++)
            if (idx[i] >= 0 && labels[idx[i]] != 0) n++;
        return n;
    }

    private static bool TopKDiffers(ReadOnlySpan<int> a, ReadOnlySpan<int> b)
    {
        Span<int> sa = stackalloc int[K];
        Span<int> sb = stackalloc int[K];
        a.CopyTo(sa); b.CopyTo(sb);
        sa.Sort(); sb.Sort();
        for (int i = 0; i < K; i++) if (sa[i] != sb[i]) return true;
        return false;
    }

    private record QueryRow(string Id, float[] Vec);

    private static List<QueryRow> LoadQueries(string path, JsonVectorizer jvec)
    {
        var bytes = File.ReadAllBytes(path);
        using var doc = JsonDocument.Parse(bytes);
        var entries = doc.RootElement.GetProperty("entries");
        var list = new List<QueryRow>(entries.GetArrayLength());
        foreach (var entry in entries.EnumerateArray())
        {
            var req = entry.GetProperty("request");
            var raw = Encoding.UTF8.GetBytes(req.GetRawText());
            var vec = new float[Dataset.Dimensions];
            jvec.VectorizeJson(raw, vec);
            string id = req.TryGetProperty("id", out var idEl) ? idEl.GetString() ?? "" : "";
            list.Add(new QueryRow(id, vec));
        }
        return list;
    }

    private static string FindRepoRoot()
    {
        var dir = AppContext.BaseDirectory;
        while (dir is not null && !File.Exists(Path.Combine(dir, "Rinha.slnx")))
            dir = Path.GetDirectoryName(dir);
        return dir ?? throw new InvalidOperationException("Could not locate repo root");
    }
}
