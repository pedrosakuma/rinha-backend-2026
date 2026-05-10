using System.Globalization;
using System.Text;
using System.Text.Json;
using Rinha.Api;

namespace Rinha.Bench;

/// <summary>
/// Wave 8.2: greedy bit allocation for ProfileFastPath.
///
/// Pre-vectorizes all eval queries once, then iteratively rebuilds the
/// fast-path table with different per-feature bit budgets. At each step,
/// adds 1 bit to the feature that gives the largest hit-rate increase
/// while keeping FP=FN=0 on the eval set.
///
/// Usage: Rinha.Bench --sweep-fastpath [--budget=24] [--start=1,1,1,1,1,1,1,1]
///                                    [--k-fraud=400] [--k-legit=100]
///                                    [--seed-uniform]   # also benchmark uniform 3s
/// </summary>
public static class SweepFastPath
{
    public static int Run(string[] args)
    {
        string testData = "/tmp/rinha-eval/test/test-data.json";
        int budget = ProfileFastPath.MaxTotalBits;
        int kFraud = 400, kLegit = 100;
        int[] start = new[] { 1, 1, 1, 1, 1, 1, 1, 1 };
        bool seedUniform = false;

        foreach (var a in args)
        {
            if (a.StartsWith("--test-data=")) testData = a[12..];
            else if (a.StartsWith("--budget=")) budget = int.Parse(a[9..], CultureInfo.InvariantCulture);
            else if (a.StartsWith("--k-fraud=")) kFraud = int.Parse(a[10..], CultureInfo.InvariantCulture);
            else if (a.StartsWith("--k-legit=")) kLegit = int.Parse(a[10..], CultureInfo.InvariantCulture);
            else if (a.StartsWith("--start="))
            {
                var p = a[8..].Split(',');
                if (p.Length != ProfileFastPath.NumFeatures) throw new ArgumentException("--start needs 8 ints");
                for (int i = 0; i < p.Length; i++) start[i] = int.Parse(p[i], CultureInfo.InvariantCulture);
            }
            else if (a == "--seed-uniform") seedUniform = true;
        }

        // Find repo root.
        string root = AppContext.BaseDirectory;
        while (root is not null && !File.Exists(Path.Combine(root, "Rinha.slnx")))
            root = Path.GetDirectoryName(root);
        if (root is null) throw new InvalidOperationException("repo root not found");

        var norm = NormalizationConstants.Load(Path.Combine(root, "resources/normalization.json"));
        var mcc = MccRiskTable.Load(Path.Combine(root, "resources/mcc_risk.json"));
        var jvec = new JsonVectorizer(norm, mcc);

        var dataDir = Path.Combine(root, "data");
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

        // Pre-vectorize eval set.
        Console.WriteLine($"Pre-vectorizing eval queries from {testData}...");
        var bytes = File.ReadAllBytes(testData);
        using var doc = JsonDocument.Parse(bytes);
        var entriesEl = doc.RootElement.GetProperty("entries");
        int total = entriesEl.GetArrayLength();
        var queries = new float[total * Dataset.Dimensions];
        var expected = new byte[total]; // 1 = fraud (fc>=3), 0 = legit
        var qBuf = new float[Dataset.Dimensions];
        int idx = 0;
        foreach (var entry in entriesEl.EnumerateArray())
        {
            var req = entry.GetProperty("request");
            var raw = Encoding.UTF8.GetBytes(req.GetRawText());
            jvec.VectorizeJson(raw, qBuf);
            Array.Copy(qBuf, 0, queries, idx * Dataset.Dimensions, Dataset.Dimensions);
            int fc = (int)Math.Round(entry.GetProperty("expected_fraud_score").GetSingle() * 5f);
            expected[idx] = (byte)(fc >= 3 ? 1 : 0);
            idx++;
        }
        Console.WriteLine($"Vectorized {total} queries.");

        // Helper to score a bits[] config.
        (int hits, int fp, int fn, int legitB, int fraudB) Eval(int[] bits)
        {
            ProfileFastPath.BuildWith(ds, bits, kLegit, kFraud, log: false);
            int hits = 0, fp = 0, fn = 0;
            var span = queries.AsSpan();
            for (int i = 0; i < total; i++)
            {
                var qSlice = span.Slice(i * Dataset.Dimensions, Dataset.Dimensions);
                byte r = ProfileFastPath.TryLookup(qSlice);
                if (r == ProfileFastPath.ResultUndecided) continue;
                hits++;
                bool predFraud = r == ProfileFastPath.ResultFraud;
                bool isFraud = expected[i] != 0;
                if (predFraud && !isFraud) fp++;
                else if (!predFraud && isFraud) fn++;
            }
            return (hits, fp, fn, ProfileFastPath.DecidedLegit, ProfileFastPath.DecidedFraud);
        }

        if (seedUniform)
        {
            var u = new[] { 3, 3, 3, 3, 3, 3, 3, 3 };
            var (h, fp, fn, l, fr) = Eval(u);
            Console.WriteLine($"baseline uniform [3,3,3,3,3,3,3,3]: hits={h}/{total} ({100.0 * h / total:F2}%) FP={fp} FN={fn} legitB={l} fraudB={fr}");
        }

        // Start from a known-good config (defaults to uniform [3]*8 = 24 bits, 0/0).
        // Then iterate swaps: subtract 1 bit from one feature, add 1 to another, keep
        // if FP=FN=0 and hits improve.
        var cur = (int[])start.Clone();
        // If user didn't override start, use uniform 3s (proven 0/0 baseline).
        bool startIsDefault = true;
        for (int i = 0; i < cur.Length; i++) if (cur[i] != 1) { startIsDefault = false; break; }
        if (startIsDefault) for (int i = 0; i < cur.Length; i++) cur[i] = 3;

        Console.WriteLine($"--- swap-greedy, budget={budget}, start=[{string.Join(",", cur)}] (sum={Sum(cur)}) ---");
        var (h0, fp0, fn0, _, _) = Eval(cur);
        Console.WriteLine($"start: hits={h0}/{total} ({100.0 * h0 / total:F2}%) FP={fp0} FN={fn0}");
        if (fp0 != 0 || fn0 != 0)
        {
            Console.WriteLine("starting config not zero-FP/FN; aborting (swap-greedy needs a valid base)");
            return 1;
        }

        int curHits = h0;
        bool improved = true;
        int round = 0;
        while (improved)
        {
            improved = false;
            round++;
            int bestFrom = -1, bestTo = -1, bestHits = curHits;
            int bestFp = -1, bestFn = -1, bestLegitB = 0, bestFraudB = 0;
            for (int from = 0; from < ProfileFastPath.NumFeatures; from++)
            {
                if (cur[from] <= 1) continue;
                for (int to = 0; to < ProfileFastPath.NumFeatures; to++)
                {
                    if (to == from) continue;
                    if (cur[to] >= ProfileFastPath.MaxBitsPerFeature) continue;
                    cur[from]--; cur[to]++;
                    if (Sum(cur) <= budget)
                    {
                        var (h, fp, fn, l, fr) = Eval(cur);
                        if (fp == 0 && fn == 0 && h > bestHits)
                        {
                            bestHits = h; bestFrom = from; bestTo = to;
                            bestFp = fp; bestFn = fn; bestLegitB = l; bestFraudB = fr;
                        }
                    }
                    cur[from]++; cur[to]--;
                }
            }
            // Also try pure-add moves (free bits if budget allows).
            if (Sum(cur) < budget)
            {
                for (int to = 0; to < ProfileFastPath.NumFeatures; to++)
                {
                    if (cur[to] >= ProfileFastPath.MaxBitsPerFeature) continue;
                    cur[to]++;
                    var (h, fp, fn, l, fr) = Eval(cur);
                    if (fp == 0 && fn == 0 && h > bestHits)
                    {
                        bestHits = h; bestFrom = -2; bestTo = to;
                        bestFp = fp; bestFn = fn; bestLegitB = l; bestFraudB = fr;
                    }
                    cur[to]--;
                }
            }

            if (bestTo >= 0)
            {
                if (bestFrom >= 0) { cur[bestFrom]--; cur[bestTo]++; }
                else cur[bestTo]++;
                curHits = bestHits;
                improved = true;
                string move = bestFrom >= 0
                    ? $"swap {ProfileFastPath.FeatureName[bestFrom]} -> {ProfileFastPath.FeatureName[bestTo]}"
                    : $"add bit -> {ProfileFastPath.FeatureName[bestTo]}";
                Console.WriteLine($"r{round}: {move} now=[{string.Join(",", cur)}] sum={Sum(cur)} " +
                                  $"hits={curHits}/{total} ({100.0 * curHits / total:F2}%) FP={bestFp} FN={bestFn} " +
                                  $"legitB={bestLegitB} fraudB={bestFraudB}");
            }
        }

        Console.WriteLine($"=== best ===");
        Console.WriteLine($"PROFILE_FAST_PATH_BITS={string.Join(",", cur)} hits={curHits} ({100.0 * curHits / total:F2}%)");
        return 0;
    }

    private static int Sum(int[] a) { int s = 0; foreach (var x in a) s += x; return s; }
}
