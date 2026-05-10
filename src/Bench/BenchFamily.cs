using System.Diagnostics;
using System.Globalization;
using System.Text;
using System.Text.Json;
using Rinha.Api;
using Rinha.Api.Scorers;

namespace Rinha.Bench;

/// <summary>
/// Per-family micro-benchmark: tight loop over a single query family for fast
/// feedback when iterating on scorer changes. Reports ns/op (best-of) plus
/// scorer-internal counters. Designed for sub-10s runs and stable single-digit
/// CV%.
///
/// Usage:
///   Rinha.Bench --bench-family --family=miss-gt0 [--iterations=20]
///                              [--warmup=5] [--limit=2000] [--print-instr]
///                              [--profile-loop=N]   # loops forever for `perf record` (blocks ≥ N seconds)
///
/// Families (case-insensitive):
///   miss-all, miss-gt0, miss-gt02, miss-gt04, miss-gt06, miss-gt08, miss-gt1,
///   miss-d2-q25, miss-d2-q50, miss-d2-q75, miss-d2-q95, miss-d2-tail,
///   hit, all
/// </summary>
public static class BenchFamily
{
    public static int Run(string[] args)
    {
        string testData = "/tmp/rinha-eval/test/test-data.json";
        string family = "miss-gt0";
        int iterations = 20;
        int warmup = 5;
        int limit = 0; // 0 = all queries in the family
        bool printInstr = false;
        int profileSeconds = 0; // > 0 = loop forever (until killed) for perf record
        foreach (var a in args)
        {
            if (a.StartsWith("--test-data=")) testData = a[12..];
            else if (a.StartsWith("--family=")) family = a[9..].ToLowerInvariant();
            else if (a.StartsWith("--iterations=")) iterations = int.Parse(a[13..], CultureInfo.InvariantCulture);
            else if (a.StartsWith("--warmup=")) warmup = int.Parse(a[9..], CultureInfo.InvariantCulture);
            else if (a.StartsWith("--limit=")) limit = int.Parse(a[8..], CultureInfo.InvariantCulture);
            else if (a == "--print-instr") printInstr = true;
            else if (a.StartsWith("--profile-loop=")) profileSeconds = int.Parse(a[15..], CultureInfo.InvariantCulture);
        }

        string root = AppContext.BaseDirectory;
        while (root is not null && !File.Exists(Path.Combine(root, "Rinha.slnx")))
            root = Path.GetDirectoryName(root);
        if (root is null) throw new InvalidOperationException("repo root not found");

        var norm = NormalizationConstants.Load(Path.Combine(root, "resources/normalization.json"));
        var mcc = MccRiskTable.Load(Path.Combine(root, "resources/mcc_risk.json"));
        var jvec = new JsonVectorizer(norm, mcc);

        var dataDir = Path.Combine(root, "data");
        Console.Error.WriteLine($"Loading dataset from {dataDir}...");
        using var ds = Dataset.Open(
            Path.Combine(dataDir, "references.bin"),
            Path.Combine(dataDir, "labels.bin"),
            File.Exists(Path.Combine(dataDir, "references_q8.bin")) ? Path.Combine(dataDir, "references_q8.bin") : null,
            File.Exists(Path.Combine(dataDir, "references_q8_soa.bin")) ? Path.Combine(dataDir, "references_q8_soa.bin") : null,
            File.Exists(Path.Combine(dataDir, "references_q16.bin")) ? Path.Combine(dataDir, "references_q16.bin") : null,
            null,
            Path.Combine(dataDir, "ivf_centroids.bin"),
            File.Exists(Path.Combine(dataDir, "ivf_offsets.bin")) ? Path.Combine(dataDir, "ivf_offsets.bin") : null,
            File.Exists(Path.Combine(dataDir, "ivf_bbox_min.bin")) ? Path.Combine(dataDir, "ivf_bbox_min.bin") : null,
            File.Exists(Path.Combine(dataDir, "ivf_bbox_max.bin")) ? Path.Combine(dataDir, "ivf_bbox_max.bin") : null,
            File.Exists(Path.Combine(dataDir, "references_q16_blocked.bin")) ? Path.Combine(dataDir, "references_q16_blocked.bin") : null,
            File.Exists(Path.Combine(dataDir, "ivf_block_offsets.bin")) ? Path.Combine(dataDir, "ivf_block_offsets.bin") : null);
        Console.Error.WriteLine($"Loaded {ds.Count} refs.");

        ProfileFastPath.Build(ds);
        var scorer = new IvfBlockedScorer(ds);

        Console.Error.WriteLine($"Loading {testData}...");
        var bytes = File.ReadAllBytes(testData);
        using var doc = JsonDocument.Parse(bytes);
        var entriesEl = doc.RootElement.GetProperty("entries");
        int total = entriesEl.GetArrayLength();
        var allVecs = new float[total][];
        var gtScore = new float[total];
        var qBuf = new float[Dataset.Dimensions];
        int idx = 0;
        foreach (var entry in entriesEl.EnumerateArray())
        {
            var req = entry.GetProperty("request");
            var raw = Encoding.UTF8.GetBytes(req.GetRawText());
            jvec.VectorizeJson(raw, qBuf);
            var v = new float[Dataset.Dimensions];
            Array.Copy(qBuf, 0, v, 0, Dataset.Dimensions);
            allVecs[idx] = v;
            gtScore[idx] = entry.GetProperty("expected_fraud_score").GetSingle();
            idx++;
        }

        // Compute fpHit and nearest centroid d² (for d²-based families).
        var fpHit = new bool[total];
        for (int q = 0; q < total; q++)
            fpHit[q] = ProfileFastPath.TryLookup(allVecs[q]) != ProfileFastPath.ResultUndecided;

        var nearestDist = new float[total];
        unsafe
        {
            float* cents = ds.CentroidsPtr;
            int nlist = ds.NumCells;
            for (int q = 0; q < total; q++)
            {
                var v = allVecs[q];
                float best = float.MaxValue;
                for (int c = 0; c < nlist; c++)
                {
                    float* cp = cents + (long)c * Dataset.PaddedDimensions;
                    float d2 = 0f;
                    for (int k = 0; k < Dataset.Dimensions; k++)
                    {
                        float diff = v[k] - cp[k];
                        d2 += diff * diff;
                    }
                    if (d2 < best) best = d2;
                }
                nearestDist[q] = best;
            }
        }

        // d² quantiles over miss-only set.
        var missDists = new List<float>();
        for (int q = 0; q < total; q++) if (!fpHit[q]) missDists.Add(nearestDist[q]);
        missDists.Sort();
        float q25 = missDists[(int)(missDists.Count * 0.25)];
        float q50 = missDists[(int)(missDists.Count * 0.50)];
        float q75 = missDists[(int)(missDists.Count * 0.75)];
        float q95 = missDists[(int)(missDists.Count * 0.95)];

        Func<int, bool> sel = family switch
        {
            "all"          => q => true,
            "hit"          => q => fpHit[q],
            "miss-all"     => q => !fpHit[q],
            "miss-gt0"     => q => !fpHit[q] && gtScore[q] < 0.1f,
            "miss-gt02"    => q => !fpHit[q] && gtScore[q] >= 0.1f && gtScore[q] < 0.3f,
            "miss-gt04"    => q => !fpHit[q] && gtScore[q] >= 0.3f && gtScore[q] < 0.5f,
            "miss-gt06"    => q => !fpHit[q] && gtScore[q] >= 0.5f && gtScore[q] < 0.7f,
            "miss-gt08"    => q => !fpHit[q] && gtScore[q] >= 0.7f && gtScore[q] < 0.9f,
            "miss-gt1"     => q => !fpHit[q] && gtScore[q] >= 0.9f,
            "miss-d2-q25"  => q => !fpHit[q] && nearestDist[q] <= q25,
            "miss-d2-q50"  => q => !fpHit[q] && nearestDist[q] > q25 && nearestDist[q] <= q50,
            "miss-d2-q75"  => q => !fpHit[q] && nearestDist[q] > q50 && nearestDist[q] <= q75,
            "miss-d2-q95"  => q => !fpHit[q] && nearestDist[q] > q75 && nearestDist[q] <= q95,
            "miss-d2-tail" => q => !fpHit[q] && nearestDist[q] > q95,
            _ => throw new ArgumentException($"unknown family '{family}'")
        };

        var familyVecs = new List<float[]>();
        for (int q = 0; q < total; q++) if (sel(q)) familyVecs.Add(allVecs[q]);
        if (limit > 0 && familyVecs.Count > limit) familyVecs = familyVecs.GetRange(0, limit);
        int n = familyVecs.Count;
        if (n == 0) { Console.Error.WriteLine("(empty family)"); return 1; }

        var arr = familyVecs.ToArray();

        Console.Error.WriteLine($"Family '{family}': n={n} queries, iterations={iterations} warmup={warmup}");

        // Profile-loop mode: spin forever calling the family in order so perf record
        // can attach. No stopwatch, no allocation in loop.
        if (profileSeconds > 0)
        {
            Console.Error.WriteLine($"PROFILE-LOOP mode: looping for >= {profileSeconds}s for perf record. Ctrl-C to stop.");
            var swDeadline = Stopwatch.StartNew();
            long iter = 0;
            float sink = 0f;
            while (swDeadline.Elapsed.TotalSeconds < profileSeconds)
            {
                for (int i = 0; i < arr.Length; i++)
                {
                    sink += scorer.Score(arr[i]);
                }
                iter++;
            }
            Console.WriteLine($"profile-loop done: {iter} passes × {arr.Length} queries in {swDeadline.Elapsed.TotalSeconds:F1}s sink={sink}");
            return 0;
        }

        // Warmup
        for (int w = 0; w < warmup; w++)
            for (int i = 0; i < arr.Length; i++) scorer.Score(arr[i]);

        // Timed iterations: each iteration scans the whole family once. Best-of, mean, CV.
        var iterTicks = new long[iterations];
        var sw = new Stopwatch();
        for (int it = 0; it < iterations; it++)
        {
            sw.Restart();
            for (int i = 0; i < arr.Length; i++) scorer.Score(arr[i]);
            sw.Stop();
            iterTicks[it] = sw.ElapsedTicks;
        }
        double tickToNs = 1_000_000_000.0 / Stopwatch.Frequency;
        Array.Sort(iterTicks);
        double bestNs = iterTicks[0] * tickToNs;
        double medianNs = iterTicks[iterations / 2] * tickToNs;
        double meanNs = 0; foreach (var t in iterTicks) meanNs += t * tickToNs; meanNs /= iterations;
        double sd = 0; foreach (var t in iterTicks) { double d = t * tickToNs - meanNs; sd += d * d; } sd = Math.Sqrt(sd / iterations);
        double cv = sd / meanNs * 100.0;

        double bestNsPerOp = bestNs / arr.Length;
        double medNsPerOp = medianNs / arr.Length;

        Console.WriteLine($"family={family} n={arr.Length} iters={iterations}");
        Console.WriteLine($"  best/op = {bestNsPerOp,8:F0} ns  ({bestNsPerOp/1000.0:F2} µs)");
        Console.WriteLine($"  med/op  = {medNsPerOp,8:F0} ns  ({medNsPerOp/1000.0:F2} µs)");
        Console.WriteLine($"  CV      = {cv,5:F1}% over {iterations} passes");

        if (printInstr)
        {
            IvfBlockedScorer.InstrumentationEnabled = true;
            IvfBlockedScorer.ResetCounters();
            for (int i = 0; i < arr.Length; i++) scorer.Score(arr[i]);
            IvfBlockedScorer.InstrumentationEnabled = false;
            double cellsScannedAvg = 4 + (double)IvfBlockedScorer.CountCellsBboxScanned / arr.Length;
            double bboxSkippedAvg = (double)IvfBlockedScorer.CountCellsBboxSkipped / arr.Length;
            double blocksAvg = (double)IvfBlockedScorer.CountBlocksConsidered / arr.Length;
            double partialPct = blocksAvg > 0 ? 100.0 * IvfBlockedScorer.CountBlocksPartialPruned / IvfBlockedScorer.CountBlocksConsidered : 0;
            double fullPrunedAvg = (double)IvfBlockedScorer.CountBlocksFullPruned / arr.Length;
            double accAvg = (double)IvfBlockedScorer.CountBlocksAccepted / arr.Length;
            double triAvg = (double)IvfBlockedScorer.CountBlocksTriPruned / arr.Length;
            Console.WriteLine($"  cells/q = {cellsScannedAvg:F1}  bbox_skipped/q = {bboxSkippedAvg:F1}");
            Console.WriteLine($"  blocks/q = {blocksAvg:F0}  partial_pruned = {partialPct:F1}%  full_pruned/q = {fullPrunedAvg:F1}  accepted/q = {accAvg:F1}  tri_pruned/q = {triAvg:F1}");
            // Bytes streamed (per query, excluding centroid table).
            double bytesPerBlock = 224.0;
            double mbPerQ = blocksAvg * bytesPerBlock / 1e6;
            Console.WriteLine($"  bandwidth ≈ {mbPerQ:F2} MB/q (≈ {mbPerQ * arr.Length / 1024:F1} GB/iteration)");
        }

        return 0;
    }
}
