using System.Diagnostics;
using System.Globalization;
using System.Text;
using System.Text.Json;
using Rinha.Api;

namespace Rinha.Bench;

/// <summary>
/// Per-stage isolated latency histogram: for each cascade stage, calls TryLookup
/// over all queries (best-of-N) and reports a fine-grained latency distribution.
/// Distinguishes stage-internal variance (cache miss on table[]) from cumulative
/// cascade variance, so we know which stage is responsible for tail outliers.
///
/// Usage: Rinha.Bench --stage-histogram [--test-data=...] [--repeats=5]
/// </summary>
public static class StageHistogram
{
    public static int Run(string[] args)
    {
        string testData = "/tmp/rinha-eval/test/test-data.json";
        int repeats = 5;
        string? dataDirArg = null;
        foreach (var a in args)
        {
            if (a.StartsWith("--test-data=")) testData = a[12..];
            else if (a.StartsWith("--repeats=")) repeats = int.Parse(a[10..], CultureInfo.InvariantCulture);
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

        var cascade = SelectiveDecisionCascade.Build(ds, Path.Combine(root, "resources/selective_decision_tables.json"));
        var stages = cascade.Stages;
        Console.WriteLine($"Cascade has {stages.Count} stage(s).");

        Console.WriteLine($"Loading {testData} ...");
        var bytes = File.ReadAllBytes(testData);
        using var doc = JsonDocument.Parse(bytes);
        var entriesEl = doc.RootElement.GetProperty("entries");
        int total = entriesEl.GetArrayLength();
        var vecs = new float[total][];
        var qBuf = new float[Dataset.Dimensions];
        int idx = 0;
        foreach (var entry in entriesEl.EnumerateArray())
        {
            var req = entry.GetProperty("request");
            var raw = Encoding.UTF8.GetBytes(req.GetRawText());
            jvec.VectorizeJson(raw, qBuf);
            var v = new float[Dataset.Dimensions];
            Array.Copy(qBuf, 0, v, 0, Dataset.Dimensions);
            vecs[idx++] = v;
        }
        Console.WriteLine($"Loaded {total} queries.");

        // Determine reaching set per stage (queries that actually invoke stage S in production).
        var reachStage = new int[total];
        for (int q = 0; q < total; q++)
        {
            cascade.TryLookupWithStage(vecs[q], out int s);
            reachStage[q] = s; // -1 if all undecided
        }

        double tickToNs = 1_000_000_000.0 / Stopwatch.Frequency;
        var sw = new Stopwatch();

        // Warmup: run all stages once over a slice to JIT and warm caches.
        for (int q = 0; q < Math.Min(5000, total); q++)
            for (int s = 0; s < stages.Count; s++)
                stages[s].TryLookup(vecs[q]);

        for (int s = 0; s < stages.Count; s++)
        {
            var stage = stages[s];
            // Time stage[s] in isolation across all queries (best-of-repeats).
            var ticks = new long[total];
            for (int q = 0; q < total; q++)
            {
                long best = long.MaxValue;
                for (int r = 0; r < repeats; r++)
                {
                    sw.Restart();
                    stage.TryLookup(vecs[q]);
                    sw.Stop();
                    long t = sw.ElapsedTicks;
                    if (t < best) best = t;
                }
                ticks[q] = best;
            }

            ReportStage($"stage[{s}] {stage.Name} (all {total} queries, isolated)", ticks, tickToNs);

            // Subset: queries that REACH this stage in production (prior stages all returned Undecided).
            var reach = new List<long>(total);
            for (int q = 0; q < total; q++) if (reachStage[q] >= s || reachStage[q] < 0) reach.Add(ticks[q]);
            ReportStage($"stage[{s}] {stage.Name} (reach-set n={reach.Count})", reach.ToArray(), tickToNs);
        }

        // Cumulative cascade time (production hot path, including all stages until decision/miss).
        var cumTicks = new long[total];
        for (int q = 0; q < total; q++)
        {
            long best = long.MaxValue;
            for (int r = 0; r < repeats; r++)
            {
                sw.Restart();
                cascade.TryLookup(vecs[q]);
                sw.Stop();
                long t = sw.ElapsedTicks;
                if (t < best) best = t;
            }
            cumTicks[q] = best;
        }
        ReportStage("CUMULATIVE cascade.TryLookup (all queries)", cumTicks, tickToNs);

        // Subset breakdowns for cumulative
        for (int s = -1; s < stages.Count; s++)
        {
            var subset = new List<long>(total);
            for (int q = 0; q < total; q++) if (reachStage[q] == s) subset.Add(cumTicks[q]);
            string label = s < 0 ? "miss (no stage decided)" : $"decided at stage[{s}] {stages[s].Name}";
            if (subset.Count > 0)
                ReportStage($"  cumulative — {label} (n={subset.Count})", subset.ToArray(), tickToNs);
        }

        return 0;
    }

    private static void ReportStage(string label, long[] ticks, double tickToNs)
    {
        if (ticks.Length == 0) { Console.WriteLine($"{label}: empty"); return; }
        Array.Sort(ticks);
        long sum = 0; foreach (var t in ticks) sum += t;
        double mean = (double)sum / ticks.Length * tickToNs;
        double p50 = ticks[(int)(ticks.Length * 0.50)] * tickToNs;
        double p90 = ticks[(int)(ticks.Length * 0.90)] * tickToNs;
        double p99 = ticks[(int)(ticks.Length * 0.99)] * tickToNs;
        double p999 = ticks[(int)(ticks.Length * 0.999)] * tickToNs;
        double max = ticks[^1] * tickToNs;
        Console.WriteLine($"{label}: n={ticks.Length} mean={mean:F0}ns p50={p50:F0}ns p90={p90:F0}ns p99={p99:F0}ns p99.9={p999:F0}ns max={max:F0}ns");
    }
}
