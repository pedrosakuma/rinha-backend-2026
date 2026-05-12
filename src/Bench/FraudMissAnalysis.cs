using System.Globalization;
using System.Text;
using System.Text.Json;
using Rinha.Api;

namespace Rinha.Bench;

/// <summary>
/// Analyzes "miss &amp; GT=1.0" cohort (fraud-clear queries that FastPath does not decide)
/// to find shared feature patterns. Tries to discover thresholds/rules that:
///   - cover a large fraction of fraud-clear misses (TPR on the fraud cohort)
///   - have ZERO or near-zero false positives on the legit-clear miss cohort (FPR≈0).
/// If such a rule exists, it can become a pre-IVF "fast-fraud" shortcut.
///
/// Usage: Rinha.Bench --fraud-miss-analysis [--test-data=...]
/// </summary>
public static class FraudMissAnalysis
{
    static readonly string[] FeatureNames =
    {
        "amt/maxAmt",    // 0
        "installments",  // 1
        "amt/custAvg",   // 2
        "hour/23",       // 3
        "dow/6",         // 4
        "min_since_last",// 5  (-1 if absent)
        "km_from_last",  // 6  (-1 if absent)
        "km_from_home",  // 7
        "tx_count_24h",  // 8
        "is_online",     // 9  (0/1)
        "card_present",  // 10 (0/1)
        "unknown_merch", // 11 (0/1)
        "mcc_risk",      // 12
        "merch_avg",     // 13
    };

    public static int Run(string[] args)
    {
        string testData = "/tmp/rinha-eval/test/test-data.json";
        foreach (var a in args)
            if (a.StartsWith("--test-data=")) testData = a[12..];

        string root = AppContext.BaseDirectory;
        while (root is not null && !File.Exists(Path.Combine(root, "Rinha.slnx")))
            root = Path.GetDirectoryName(root);
        if (root is null) throw new InvalidOperationException("repo root not found");

        var norm = NormalizationConstants.Load(Path.Combine(root, "resources/normalization.json"));
        var mcc = MccRiskTable.Load(Path.Combine(root, "resources/mcc_risk.json"));
        var jvec = new JsonVectorizer(norm, mcc);

        // Need dataset only to build the FastPath cache (so we know which queries miss).
        var dataDir = Path.Combine(root, "data");
        using var ds = Dataset.Open(
            Path.Combine(dataDir, "references.bin"),
            Path.Combine(dataDir, "labels.bin"),
            null, null, null, null,
            Path.Combine(dataDir, "ivf_centroids.bin"), null, null, null, null, null);
        ProfileFastPath.Build(ds);

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

        // Partition: miss-only cohorts.
        var fraudMiss = new List<int>();    // GT >= 0.9 AND FastPath miss
        var legitMiss = new List<int>();    // GT <= 0.1 AND FastPath miss
        var allMiss   = new List<int>();
        for (int q = 0; q < total; q++)
        {
            bool fpHit = ProfileFastPath.TryLookup(vecs[q]) != ProfileFastPath.ResultUndecided;
            if (fpHit) continue;
            allMiss.Add(q);
            if (gtScore[q] >= 0.9f) fraudMiss.Add(q);
            else if (gtScore[q] <= 0.1f) legitMiss.Add(q);
        }
        Console.WriteLine($"Total queries: {total}");
        Console.WriteLine($"Total FastPath misses: {allMiss.Count}");
        Console.WriteLine($"  - fraud-clear (GT≥0.9): {fraudMiss.Count}");
        Console.WriteLine($"  - legit-clear (GT≤0.1): {legitMiss.Count}");

        // -------- Per-feature stats: mean and 5/25/50/75/95 percentile on fraud cohort
        Console.WriteLine();
        Console.WriteLine("=== Feature distribution: fraud-miss vs legit-miss ===");
        Console.WriteLine($"{"feat",-16} {"fraud_mean",10} {"legit_mean",10} {"f_p25",8} {"f_p50",8} {"f_p75",8} {"l_p25",8} {"l_p50",8} {"l_p75",8}");
        for (int f = 0; f < Dataset.Dimensions; f++)
        {
            var fVals = fraudMiss.Select(q => vecs[q][f]).OrderBy(x => x).ToArray();
            var lVals = legitMiss.Select(q => vecs[q][f]).OrderBy(x => x).ToArray();
            double fMean = fVals.Average();
            double lMean = lVals.Average();
            float Fp(float[] a, double p) => a[(int)(a.Length * p)];
            Console.WriteLine($"{FeatureNames[f],-16} {fMean,10:F4} {lMean,10:F4} {Fp(fVals,0.25),8:F4} {Fp(fVals,0.50),8:F4} {Fp(fVals,0.75),8:F4} {Fp(lVals,0.25),8:F4} {Fp(lVals,0.50),8:F4} {Fp(lVals,0.75),8:F4}");
        }

        // -------- Single-feature threshold sweep
        // For each feature & direction (≥T or ≤T), find threshold that maximizes
        //   coverage of fraud-miss subject to FPR on legit-miss < 0.5% (≤3 FPs).
        Console.WriteLine();
        Console.WriteLine("=== Single-feature rule sweep (target: ≥X% fraud-miss coverage, ≤3 legit-miss FP) ===");
        Console.WriteLine($"{"rule",-40} {"tpr",8} {"fp(legit)",10}");
        for (int f = 0; f < Dataset.Dimensions; f++)
        {
            // Try ≥T (high values predict fraud)
            BestRule(f, vecs, fraudMiss, legitMiss, geq: true);
            BestRule(f, vecs, fraudMiss, legitMiss, geq: false);
        }

        // -------- Pairwise rule search (greedy 2-feature AND)
        Console.WriteLine();
        Console.WriteLine("=== 2-feature AND rules (target: TPR maximal, legit-miss FP ≤ 3) ===");
        var topRules = new List<(string desc, int tp, int fp)>();
        for (int f1 = 0; f1 < Dataset.Dimensions; f1++)
        {
            for (int f2 = f1 + 1; f2 < Dataset.Dimensions; f2++)
            {
                foreach (var d1 in new[] { true, false })
                foreach (var d2 in new[] { true, false })
                {
                    var (desc, tp, fp) = BestPairRule(f1, f2, d1, d2, vecs, fraudMiss, legitMiss);
                    if (tp >= 50 && fp <= 3) topRules.Add((desc, tp, fp));
                }
            }
        }
        topRules.Sort((a, b) => b.tp.CompareTo(a.tp));
        foreach (var (desc, tp, fp) in topRules.Take(20))
            Console.WriteLine($"  {desc,-60} TPR={tp}/{fraudMiss.Count} ({100.0*tp/fraudMiss.Count:F1}%)  FP={fp}");

        // -------- Coverage stacking: greedily build a union of rules
        Console.WriteLine();
        Console.WriteLine("=== Greedy union of 2-feature rules (FP budget: 5) ===");
        StackRules(vecs, fraudMiss, legitMiss);

        return 0;
    }

    static void BestRule(int f, float[][] vecs, List<int> fraud, List<int> legit, bool geq)
    {
        // Collect candidate thresholds from fraud cohort (sorted).
        var ts = fraud.Select(q => vecs[q][f]).Distinct().OrderBy(x => x).ToArray();
        int bestTp = 0; int bestFp = int.MaxValue; float bestT = 0;
        foreach (var t in ts)
        {
            int tp = 0, fp = 0;
            foreach (var q in fraud) if (geq ? vecs[q][f] >= t : vecs[q][f] <= t) tp++;
            foreach (var q in legit) if (geq ? vecs[q][f] >= t : vecs[q][f] <= t) fp++;
            if (fp <= 3 && tp > bestTp) { bestTp = tp; bestFp = fp; bestT = t; }
        }
        if (bestTp >= 20)
        {
            string op = geq ? "≥" : "≤";
            Console.WriteLine($"  {FeatureNames[f]} {op} {bestT,7:F4}                                {bestTp,3}/{fraud.Count} ({100.0*bestTp/fraud.Count:F1}%)  FP={bestFp}");
        }
    }

    static (string desc, int tp, int fp) BestPairRule(
        int f1, int f2, bool geq1, bool geq2,
        float[][] vecs, List<int> fraud, List<int> legit)
    {
        // Get candidate thresholds (sample from fraud cohort percentiles).
        var t1s = SampleThresholds(fraud.Select(q => vecs[q][f1]));
        var t2s = SampleThresholds(fraud.Select(q => vecs[q][f2]));
        int bestTp = 0, bestFp = int.MaxValue;
        float bt1 = 0, bt2 = 0;
        foreach (var t1 in t1s)
        foreach (var t2 in t2s)
        {
            int tp = 0, fp = 0;
            foreach (var q in fraud)
            {
                var v = vecs[q];
                if ((geq1 ? v[f1] >= t1 : v[f1] <= t1) && (geq2 ? v[f2] >= t2 : v[f2] <= t2)) tp++;
            }
            foreach (var q in legit)
            {
                var v = vecs[q];
                if ((geq1 ? v[f1] >= t1 : v[f1] <= t1) && (geq2 ? v[f2] >= t2 : v[f2] <= t2)) fp++;
            }
            if (fp <= 3 && tp > bestTp) { bestTp = tp; bestFp = fp; bt1 = t1; bt2 = t2; }
        }
        string op1 = geq1 ? "≥" : "≤";
        string op2 = geq2 ? "≥" : "≤";
        return ($"{FeatureNames[f1]}{op1}{bt1:F3} AND {FeatureNames[f2]}{op2}{bt2:F3}", bestTp, bestFp);
    }

    static float[] SampleThresholds(IEnumerable<float> values)
    {
        var arr = values.ToArray();
        Array.Sort(arr);
        if (arr.Length == 0) return Array.Empty<float>();
        // 21 evenly-spaced quantiles
        var res = new float[21];
        for (int i = 0; i < 21; i++) res[i] = arr[Math.Min(arr.Length - 1, i * arr.Length / 20)];
        return res.Distinct().ToArray();
    }

    static void StackRules(float[][] vecs, List<int> fraud, List<int> legit)
    {
        var covered = new HashSet<int>();
        int fpBudget = 5;
        int fpUsed = 0;
        for (int round = 0; round < 8; round++)
        {
            (string desc, int tp, int fp, int[] coveredNow) best = ("", 0, int.MaxValue, Array.Empty<int>());
            for (int f1 = 0; f1 < Dataset.Dimensions; f1++)
            for (int f2 = f1 + 1; f2 < Dataset.Dimensions; f2++)
            foreach (var d1 in new[] { true, false })
            foreach (var d2 in new[] { true, false })
            {
                var t1s = SampleThresholds(fraud.Select(q => vecs[q][f1]));
                var t2s = SampleThresholds(fraud.Select(q => vecs[q][f2]));
                foreach (var t1 in t1s)
                foreach (var t2 in t2s)
                {
                    var hits = new List<int>();
                    int fp = 0;
                    foreach (var q in fraud)
                    {
                        var v = vecs[q];
                        if ((d1 ? v[f1] >= t1 : v[f1] <= t1) && (d2 ? v[f2] >= t2 : v[f2] <= t2))
                            if (!covered.Contains(q)) hits.Add(q);
                    }
                    foreach (var q in legit)
                    {
                        var v = vecs[q];
                        if ((d1 ? v[f1] >= t1 : v[f1] <= t1) && (d2 ? v[f2] >= t2 : v[f2] <= t2)) fp++;
                    }
                    if (fpUsed + fp > fpBudget) continue;
                    if (hits.Count > best.tp)
                    {
                        string op1 = d1 ? "≥" : "≤", op2 = d2 ? "≥" : "≤";
                        best = ($"{FeatureNames[f1]}{op1}{t1:F3} AND {FeatureNames[f2]}{op2}{t2:F3}", hits.Count, fp, hits.ToArray());
                    }
                }
            }
            if (best.tp < 20) break;
            foreach (var q in best.coveredNow) covered.Add(q);
            fpUsed += best.fp;
            Console.WriteLine($"  Round {round+1}: {best.desc,-60} +{best.tp} new TP, +{best.fp} FP, cum={covered.Count}/{fraud.Count} ({100.0*covered.Count/fraud.Count:F1}%), FP={fpUsed}/{fpBudget}");
        }
    }
}
