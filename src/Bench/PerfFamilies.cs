using System.Diagnostics;
using System.Globalization;
using System.Text;
using System.Text.Json;
using Rinha.Api;
using Rinha.Api.Scorers;

namespace Rinha.Bench;

/// <summary>
/// Per-query latency profiling that categorizes queries by attributes (fast-path
/// hit, GT score bucket, nearest-centroid distance, query feature bucket) and
/// reports p50/p95/p99/max + total CPU time per category. Identifies the biggest
/// p99 offenders so we know which family to attack next.
///
/// Usage: Rinha.Bench --perf-families [--test-data=...] [--repeats=3] [--csv=path]
/// </summary>
public static class PerfFamilies
{
    public static int Run(string[] args)
    {
        string testData = "/tmp/rinha-eval/test/test-data.json";
        int repeats = 3;
        string? csvPath = null;
        string? dataDirArg = null;
        foreach (var a in args)
        {
            if (a.StartsWith("--test-data=")) testData = a[12..];
            else if (a.StartsWith("--repeats=")) repeats = int.Parse(a[10..], CultureInfo.InvariantCulture);
            else if (a.StartsWith("--csv=")) csvPath = a[6..];
            else if (a.StartsWith("--data-dir=")) dataDirArg = a[11..];
        }

        string root = AppContext.BaseDirectory;
        while (root is not null && !File.Exists(Path.Combine(root, "Rinha.slnx")))
            root = Path.GetDirectoryName(root);
        if (root is null) throw new InvalidOperationException("repo root not found");

        var norm = NormalizationConstants.Load(Path.Combine(root, "resources/normalization.json"));
        var mcc = MccRiskTable.Load(Path.Combine(root, "resources/mcc_risk.json"));
        var jvec = new JsonVectorizer(norm, mcc);

        var dataDir = dataDirArg ?? Path.Combine(root, "data");
        Console.WriteLine($"Loading dataset from {dataDir}...");
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
        Console.WriteLine($"Loaded {ds.Count} refs.");

        var cascade = SelectiveDecisionCascade.Build(ds, Path.Combine(root, "resources/selective_decision_tables.json"));
        var stages = cascade.Stages;

        int nProbeOverride = 2;
        foreach (var a in args)
            if (a.StartsWith("--nprobe=")) nProbeOverride = int.Parse(a[9..], CultureInfo.InvariantCulture);
        Console.WriteLine($"Scorer: IvfBlockedScorer(nProbe={nProbeOverride})");
        var scorer = new IvfBlockedScorer(ds, nProbeOverride);

        Console.WriteLine($"Loading {testData} ...");
        var bytes = File.ReadAllBytes(testData);
        using var doc = JsonDocument.Parse(bytes);
        var entriesEl = doc.RootElement.GetProperty("entries");
        int total = entriesEl.GetArrayLength();
        var vecs = new float[total][];
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
            vecs[idx] = v;
            gtScore[idx] = entry.GetProperty("expected_fraud_score").GetSingle();
            idx++;
        }
        Console.WriteLine($"Loaded {total} queries.");

        // Pre-compute nearest-centroid distance² per query (proxy for IVF tail).
        Console.WriteLine("Computing nearest-centroid distance per query...");
        var nearestDist = new float[total];
        unsafe
        {
            float* cents = ds.CentroidsPtr;
            int nlist = ds.NumCells;
            for (int q = 0; q < total; q++)
            {
                var v = vecs[q];
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

        // Categorize each query.
        // a) Selective decision stage: -1=miss, otherwise index into cascade.Stages.
        var fpStage = new int[total];
        Array.Fill(fpStage, -1);
        for (int q = 0; q < total; q++)
        {
            byte result = cascade.TryLookupWithStage(vecs[q], out int stageIndex);
            if (result != SelectiveDecisionCascade.ResultUndecided)
                fpStage[q] = stageIndex;
        }

        // b) GT score bucket: 0/0.2/0.4/0.6/0.8/1.0
        // c) Nearest-centroid quartile (over MISS-only set, since hits don't run IVF)
        var missDists = new List<float>();
        for (int q = 0; q < total; q++) if (fpStage[q] == 0) missDists.Add(nearestDist[q]);
        missDists.Sort();
        float q25 = missDists[(int)(missDists.Count * 0.25)];
        float q50 = missDists[(int)(missDists.Count * 0.50)];
        float q75 = missDists[(int)(missDists.Count * 0.75)];
        float q95 = missDists[(int)(missDists.Count * 0.95)];

        // Time each query (best-of-`repeats` to reduce TP/jitter noise).
        // MIRROR PRODUCTION PIPELINE: SelectiveDecisionCascade -> IVF.
        Console.WriteLine($"Timing {total} queries × {repeats} repeats (best-of), production pipeline (selective-decision-cascade→IVF)...");
        var perQueryTicks = new long[total];
        var sw = new Stopwatch();

        // Warmup
        for (int q = 0; q < Math.Min(2000, total); q++)
        {
            if (cascade.TryLookup(vecs[q]) == SelectiveDecisionCascade.ResultUndecided)
                scorer.Score(vecs[q]);
        }

        for (int q = 0; q < total; q++)
        {
            long best = long.MaxValue;
            for (int r = 0; r < repeats; r++)
            {
                sw.Restart();
                if (cascade.TryLookup(vecs[q]) == SelectiveDecisionCascade.ResultUndecided)
                    scorer.Score(vecs[q]);
                sw.Stop();
                long t = sw.ElapsedTicks;
                if (t < best) best = t;
            }
            perQueryTicks[q] = best;
        }
        double tickToUs = 1_000_000.0 / Stopwatch.Frequency;

        // Aggregate by category.
        // Family 1: Selective decision stage
        var stageGroups = new List<(string label, Func<int, bool> sel)>();
        for (int i = 0; i < stages.Count; i++)
        {
            int stage = i;
            stageGroups.Add(($"{stages[i].Name} hit", q => fpStage[q] == stage));
        }
        stageGroups.Add(("miss", q => fpStage[q] < 0));
        ReportFamily("Selective decision stage", stageGroups.ToArray(), perQueryTicks, tickToUs);

        // Family 2: GT score bucket
        ReportFamily("GT score", new (string, Func<int, bool>)[]
        {
            ("0.0 (legit-clear)", q => gtScore[q] < 0.1f),
            ("0.2 (legit-soft)",  q => gtScore[q] >= 0.1f && gtScore[q] < 0.3f),
            ("0.4 (border-low)",  q => gtScore[q] >= 0.3f && gtScore[q] < 0.5f),
            ("0.6 (border-high)", q => gtScore[q] >= 0.5f && gtScore[q] < 0.7f),
            ("0.8 (fraud-soft)",  q => gtScore[q] >= 0.7f && gtScore[q] < 0.9f),
            ("1.0 (fraud-clear)", q => gtScore[q] >= 0.9f),
        }, perQueryTicks, tickToUs);

        // Family 3: nearest-centroid distance (only meaningful for misses; hits short-circuit)
        ReportFamily("Nearest centroid (miss-only)", new (string, Func<int, bool>)[]
        {
            ("miss & d2≤Q25",       q => fpStage[q] < 0 && nearestDist[q] <= q25),
            ("miss & Q25<d2≤Q50",   q => fpStage[q] < 0 && nearestDist[q] > q25 && nearestDist[q] <= q50),
            ("miss & Q50<d2≤Q75",   q => fpStage[q] < 0 && nearestDist[q] > q50 && nearestDist[q] <= q75),
            ("miss & Q75<d2≤Q95",   q => fpStage[q] < 0 && nearestDist[q] > q75 && nearestDist[q] <= q95),
            ("miss & d2>Q95 (tail)", q => fpStage[q] < 0 && nearestDist[q] > q95),
        }, perQueryTicks, tickToUs);

        // Family 4: GT score within MISS only (since hits dominate the "easy" buckets)
        ReportFamily("GT score (miss-only)", new (string, Func<int, bool>)[]
        {
            ("miss & 0.0", q => fpStage[q] < 0 && gtScore[q] < 0.1f),
            ("miss & 0.2", q => fpStage[q] < 0 && gtScore[q] >= 0.1f && gtScore[q] < 0.3f),
            ("miss & 0.4", q => fpStage[q] < 0 && gtScore[q] >= 0.3f && gtScore[q] < 0.5f),
            ("miss & 0.6", q => fpStage[q] < 0 && gtScore[q] >= 0.5f && gtScore[q] < 0.7f),
            ("miss & 0.8", q => fpStage[q] < 0 && gtScore[q] >= 0.7f && gtScore[q] < 0.9f),
            ("miss & 1.0", q => fpStage[q] < 0 && gtScore[q] >= 0.9f),
        }, perQueryTicks, tickToUs);

        // Top 1% offenders — what's special about them?
        var sortedIdx = Enumerable.Range(0, total).OrderByDescending(q => perQueryTicks[q]).ToArray();
        int topN = total / 100;
        Console.WriteLine();
        Console.WriteLine($"=== Top 1% slowest ({topN} queries) ===");
        int topHits = 0; double topD2 = 0; double[] topGt = new double[6];
        for (int i = 0; i < topN; i++)
        {
            int q = sortedIdx[i];
            if (fpStage[q] >= 0) topHits++;
            topD2 += nearestDist[q];
            int b = (int)Math.Round(gtScore[q] * 5f);
            if (b >= 0 && b <= 5) topGt[b == 1 || b == 0 ? Math.Min(b, 5) : b]++;
        }
        Console.WriteLine($"FastPath hits in top1%: {topHits}/{topN} ({100.0*topHits/topN:F1}%)");
        Console.WriteLine($"avg nearest_d2 in top1%: {topD2/topN:F4} (vs overall {nearestDist.Average():F4})");
        Console.Write($"GT score histogram top1%: ");
        for (int b = 0; b <= 5; b++) Console.Write($"{b/5.0:F1}={topGt[b]:F0} ");
        Console.WriteLine();

        // ===== Instrumentation: per-query scanned candidate count =====
        // Mirror IvfBlockedScorer's cell selection: pick top-nProbe cells by
        // centroid distance, sum vec_count of selected cells (using CellOffsetsPtr).
        Console.WriteLine($"Computing per-query scanned-vec count (nProbe={nProbeOverride})...");
        var scannedVecs = new int[total];
        unsafe
        {
            float* cents = ds.CentroidsPtr;
            int* offs = ds.CellOffsetsPtr;
            int nlist = ds.NumCells;
            int n = nProbeOverride;
            Span<int> cells = stackalloc int[n];
            Span<float> cellsDist = stackalloc float[n];
            for (int q = 0; q < total; q++)
            {
                var v = vecs[q];
                for (int i = 0; i < n; i++) { cells[i] = -1; cellsDist[i] = float.MaxValue; }
                float worst = float.MaxValue;
                for (int c = 0; c < nlist; c++)
                {
                    float* cp = cents + (long)c * Dataset.PaddedDimensions;
                    float d = 0f;
                    for (int k = 0; k < Dataset.Dimensions; k++) { float diff = v[k] - cp[k]; d += diff * diff; }
                    if (d < worst)
                    {
                        int pos = n - 1;
                        while (pos > 0 && cellsDist[pos - 1] > d) pos--;
                        for (int j = n - 1; j > pos; j--) { cellsDist[j] = cellsDist[j - 1]; cells[j] = cells[j - 1]; }
                        cellsDist[pos] = d; cells[pos] = c; worst = cellsDist[n - 1];
                    }
                }
                int sumVec = 0;
                for (int i = 0; i < n; i++) if (cells[i] >= 0) sumVec += offs[cells[i] + 1] - offs[cells[i]];
                scannedVecs[q] = sumVec;
            }
        }

        // Family 5: scanned-vec quartile (only meaningful for misses since hits skip IVF).
        var missScans = new List<int>();
        for (int q = 0; q < total; q++) if (fpStage[q] < 0) missScans.Add(scannedVecs[q]);
        missScans.Sort();
        int sQ25 = missScans[(int)(missScans.Count * 0.25)];
        int sQ50 = missScans[(int)(missScans.Count * 0.50)];
        int sQ75 = missScans[(int)(missScans.Count * 0.75)];
        int sQ95 = missScans[(int)(missScans.Count * 0.95)];
        Console.WriteLine($"Scanned-vec percentiles (miss-only): Q25={sQ25} Q50={sQ50} Q75={sQ75} Q95={sQ95} max={missScans[^1]}");
        ReportFamily("Scanned vectors (miss-only)", new (string, Func<int, bool>)[]
        {
            ("miss & scan≤Q25",       q => fpStage[q] < 0 && scannedVecs[q] <= sQ25),
            ("miss & Q25<scan≤Q50",   q => fpStage[q] < 0 && scannedVecs[q] > sQ25 && scannedVecs[q] <= sQ50),
            ("miss & Q50<scan≤Q75",   q => fpStage[q] < 0 && scannedVecs[q] > sQ50 && scannedVecs[q] <= sQ75),
            ("miss & Q75<scan≤Q95",   q => fpStage[q] < 0 && scannedVecs[q] > sQ75 && scannedVecs[q] <= sQ95),
            ("miss & scan>Q95 (heavy)", q => fpStage[q] < 0 && scannedVecs[q] > sQ95),
        }, perQueryTicks, tickToUs);

        // Top 1% by latency: average scanned-vec count
        double topAvgScan = 0; double overallAvgScan = scannedVecs.Where((_, i) => fpStage[i] < 0).Average();
        for (int i = 0; i < topN; i++) topAvgScan += scannedVecs[sortedIdx[i]];
        topAvgScan /= topN;
        Console.WriteLine($"avg scanned_vecs in top1%: {topAvgScan:F0} (vs miss-overall {overallAvgScan:F0}, ratio {topAvgScan/overallAvgScan:F2}x)");

        // Optional CSV dump.
        if (csvPath is not null)
        {
            using var sw2 = new StreamWriter(csvPath);
            sw2.WriteLine("idx,latency_us,fastpath_stage,gt_score,nearest_d2,scanned_vecs");
            for (int q = 0; q < total; q++)
                sw2.WriteLine($"{q},{perQueryTicks[q]*tickToUs:F2},{fpStage[q]},{gtScore[q]:F2},{nearestDist[q]:F6},{scannedVecs[q]}");
            Console.WriteLine($"CSV written to {csvPath}");
        }

        // ===== Instrumented scorer-internal counters per family =====
        // Disabled: instrumentation hooks not present in current scorer build.

        return 0;
    }

    private static void ReportFamily(string title, (string label, Func<int, bool> sel)[] groups, long[] ticks, double tickToUs)
    {
        Console.WriteLine();
        Console.WriteLine($"=== {title} ===");
        Console.WriteLine($"{"group",-32} {"n",7} {"%",6} {"p50µs",8} {"p95µs",8} {"p99µs",8} {"maxµs",8} {"sumMs",10} {"%cpu",7}");
        long totalSum = ticks.Sum();
        int totalN = ticks.Length;
        foreach (var (label, sel) in groups)
        {
            var t = new List<long>();
            long sum = 0;
            for (int q = 0; q < totalN; q++)
                if (sel(q)) { t.Add(ticks[q]); sum += ticks[q]; }
            if (t.Count == 0) { Console.WriteLine($"{label,-32} (empty)"); continue; }
            t.Sort();
            double p50 = t[(int)(t.Count * 0.50)] * tickToUs;
            double p95 = t[(int)(t.Count * 0.95)] * tickToUs;
            int idx99 = Math.Min(t.Count - 1, (int)(t.Count * 0.99));
            double p99 = t[idx99] * tickToUs;
            double mx  = t[t.Count - 1] * tickToUs;
            double sumMs = sum * tickToUs / 1000.0;
            double pctN = 100.0 * t.Count / totalN;
            double pctCpu = 100.0 * sum / totalSum;
            Console.WriteLine($"{label,-32} {t.Count,7} {pctN,5:F1}% {p50,8:F1} {p95,8:F1} {p99,8:F1} {mx,8:F1} {sumMs,10:F1} {pctCpu,6:F1}%");
        }
    }
}
