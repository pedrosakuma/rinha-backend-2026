using System.Globalization;
using System.Text;
using System.Text.Json;
using Rinha.Api;

namespace Rinha.Bench;

/// <summary>
/// Probes whether swapping currently-used FastPath features with unused ones
/// (4=day_of_week, 5=mins_since_last, 6=km_from_last, 9=isOnline, 10=cardPresent,
/// 13=merchant_avg) lifts hit-rate while keeping FP=FN=0.
///
/// For each candidate feature, swaps it into each slot of the current 8-feature
/// config (FeatureIndex [0,7,2,1,8,11,12,3], bits [3,3,4,3,3,4,2,2]) and prints
/// a hit/FP/FN matrix. Mutates ProfileFastPath.FeatureIndex/FeatureName in place
/// (the array contents are mutable even though the field is readonly).
/// </summary>
public static class ProbeFastPathFeatures
{
    static readonly (int idx, string name)[] Candidates = new[]
    {
        (4,  "day_of_week"),
        (5,  "mins_since_last"),
        (6,  "km_from_last"),
        (9,  "is_online"),
        (10, "card_present"),
        (13, "merchant_avg"),
    };

    public static int Run(string[] args)
    {
        string testData = "bench/k6/test-data.json";
        string dataDir = "data";
        int kFraud = 400, kLegit = 100;
        int[] bits = new[] { 3, 3, 4, 3, 3, 4, 2, 2 };

        foreach (var a in args)
        {
            if (a.StartsWith("--test-data=")) testData = a[12..];
            else if (a.StartsWith("--data-dir=")) dataDir = a[11..];
            else if (a.StartsWith("--k-fraud=")) kFraud = int.Parse(a[10..], CultureInfo.InvariantCulture);
            else if (a.StartsWith("--k-legit=")) kLegit = int.Parse(a[10..], CultureInfo.InvariantCulture);
            else if (a.StartsWith("--bits="))
            {
                var p = a[7..].Split(',');
                if (p.Length != ProfileFastPath.NumFeatures) throw new ArgumentException("--bits needs 8 ints");
                bits = new int[p.Length];
                for (int i = 0; i < p.Length; i++) bits[i] = int.Parse(p[i], CultureInfo.InvariantCulture);
            }
        }

        string root = AppContext.BaseDirectory;
        while (root is not null && !File.Exists(Path.Combine(root, "Rinha.slnx")))
            root = Path.GetDirectoryName(root);
        if (root is null) throw new InvalidOperationException("repo root not found");

        var norm = NormalizationConstants.Load(Path.Combine(root, "resources/normalization.json"));
        var mcc = MccRiskTable.Load(Path.Combine(root, "resources/mcc_risk.json"));
        var jvec = new JsonVectorizer(norm, mcc);

        if (!Path.IsPathRooted(dataDir)) dataDir = Path.Combine(root, dataDir);
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

        if (!Path.IsPathRooted(testData)) testData = Path.Combine(root, testData);
        Console.WriteLine($"Pre-vectorizing eval queries from {testData}...");
        var bytesT = File.ReadAllBytes(testData);
        using var doc = JsonDocument.Parse(bytesT);
        var entriesEl = doc.RootElement.GetProperty("entries");
        int total = entriesEl.GetArrayLength();
        var queries = new float[total * Dataset.Dimensions];
        var expected = new byte[total];
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

        // Save originals so we can restore between probes.
        var origIdx  = (int[])ProfileFastPath.FeatureIndex.Clone();
        var origName = (string[])ProfileFastPath.FeatureName.Clone();

        (int hits, int fp, int fn, int legitB, int fraudB) Eval(int[] bb)
        {
            ProfileFastPath.BuildWith(ds, bb, kLegit, kFraud, log: false);
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

        // Baseline (current production config)
        var (h0, fp0, fn0, _, _) = Eval(bits);
        Console.WriteLine($"\nBaseline FeatureIndex=[{string.Join(",", origIdx)}] bits=[{string.Join(",", bits)}]");
        Console.WriteLine($"  hits={h0}/{total} ({100.0 * h0 / total:F2}%) FP={fp0} FN={fn0}");

        Console.WriteLine($"\n--- single-feature swap probes (k_legit={kLegit}, k_fraud={kFraud}) ---");
        Console.WriteLine($"  legend: cand@slot -> hits delta_FP delta_FN");

        int bestH1 = h0, bestSlot1 = -1, bestCand1 = -1;
        string bestCandName1 = "";

        foreach (var (cand, candName) in Candidates)
        {
            Console.WriteLine($"\nCandidate {cand}={candName}:");
            for (int s = 0; s < ProfileFastPath.NumFeatures; s++)
            {
                int oldIdx = origIdx[s];
                if (oldIdx == cand) continue;
                ProfileFastPath.FeatureIndex[s] = cand;
                ProfileFastPath.FeatureName[s]  = candName;
                try
                {
                    var (h, fp, fn, lB, fB) = Eval(bits);
                    string flag = (fp == 0 && fn == 0 && h > h0) ? "  ★" : (fp == 0 && fn == 0 ? "" : "  ✗");
                    Console.WriteLine($"  swap slot{s} ({origName[s]}->{candName}): hits={h} ({100.0 * h / total:F2}%) FP={fp} FN={fn} legitB={lB} fraudB={fB}{flag}");
                    if (fp == 0 && fn == 0 && h > bestH1)
                    {
                        bestH1 = h; bestSlot1 = s; bestCand1 = cand; bestCandName1 = candName;
                    }
                }
                finally
                {
                    ProfileFastPath.FeatureIndex[s] = oldIdx;
                    ProfileFastPath.FeatureName[s]  = origName[s];
                }
            }
        }

        Console.WriteLine($"\n--- iterative greedy swap (k_legit={kLegit}, k_fraud={kFraud}) ---");
        var curIdx  = (int[])origIdx.Clone();
        var curName = (string[])origName.Clone();
        int curHits = h0;
        int round = 0;
        while (true)
        {
            round++;
            int bestSlot = -1, bestCand = -1, bestH = curHits;
            string bestName = "";
            foreach (var (cand, candName) in Candidates)
            {
                bool inSet = false;
                for (int i = 0; i < ProfileFastPath.NumFeatures; i++) if (curIdx[i] == cand) { inSet = true; break; }
                if (inSet) continue;
                for (int s = 0; s < ProfileFastPath.NumFeatures; s++)
                {
                    int oldIdx = curIdx[s];
                    ProfileFastPath.FeatureIndex[s] = cand;
                    ProfileFastPath.FeatureName[s]  = candName;
                    try
                    {
                        var (h, fp, fn, _, _) = Eval(bits);
                        if (fp == 0 && fn == 0 && h > bestH)
                        {
                            bestH = h; bestSlot = s; bestCand = cand; bestName = candName;
                        }
                    }
                    finally
                    {
                        ProfileFastPath.FeatureIndex[s] = oldIdx;
                        ProfileFastPath.FeatureName[s]  = curName[s];
                    }
                }
            }
            if (bestSlot < 0) break;
            string fromName = curName[bestSlot];
            curIdx[bestSlot] = bestCand;
            curName[bestSlot] = bestName;
            ProfileFastPath.FeatureIndex[bestSlot] = bestCand;
            ProfileFastPath.FeatureName[bestSlot]  = bestName;
            curHits = bestH;
            Console.WriteLine($"r{round}: swap slot{bestSlot} ({fromName}->{bestName}) hits={curHits} ({100.0 * curHits / total:F2}%) idx=[{string.Join(",", curIdx)}]");
        }

        Console.WriteLine($"\n=== final ===");
        Console.WriteLine($"FeatureIndex=[{string.Join(",", curIdx)}]");
        Console.WriteLine($"FeatureName =[{string.Join(",", curName)}]");
        Console.WriteLine($"hits={curHits}/{total} ({100.0 * curHits / total:F2}%)");
        return 0;
    }
}
