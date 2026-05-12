using System.Globalization;
using System.Text;
using System.Text.Json;
using Rinha.Api;

namespace Rinha.Bench;

/// <summary>
/// Standalone audit: runs JsonVectorizer + ProfileFastPath/2 against test-data.json
/// using the exact production fast-path order, reports FP/FN attributable to the
/// fast-paths, and prints the offending entries. Usage: --audit-fastpath [--min-count=100]
/// </summary>
public static class AuditFastPath
{
    public static int Run(string[] args)
    {
        string testData = "/tmp/rinha-eval/test/test-data.json";
        int minCount = 100;
        foreach (var a in args)
        {
            if (a.StartsWith("--test-data=")) testData = a[12..];
            else if (a.StartsWith("--min-count=")) minCount = int.Parse(a[12..], CultureInfo.InvariantCulture);
        }

        string root = AppContext.BaseDirectory;
        while (root is not null && !File.Exists(Path.Combine(root, "Rinha.slnx")))
            root = Path.GetDirectoryName(root);
        if (root is null) throw new InvalidOperationException("repo root not found");

        var norm = NormalizationConstants.Load(Path.Combine(root, "resources/normalization.json"));
        var mcc = MccRiskTable.Load(Path.Combine(root, "resources/mcc_risk.json"));
        var jvec = new JsonVectorizer(norm, mcc);

        var dataDir = Path.Combine(root, "data");

        Console.WriteLine($"Loading dataset from {dataDir}...");
        var vec = Path.Combine(dataDir, "references.bin");
        var lab = Path.Combine(dataDir, "labels.bin");
        var q8 = Path.Combine(dataDir, "references_q8.bin");
        var q8Soa = Path.Combine(dataDir, "references_q8_soa.bin");
        var q16 = Path.Combine(dataDir, "references_q16.bin");
        var q16Blocked = Path.Combine(dataDir, "references_q16_blocked.bin");
        var blockOffs = Path.Combine(dataDir, "ivf_block_offsets.bin");
        var cents = Path.Combine(dataDir, "ivf_centroids.bin");
        var offs = Path.Combine(dataDir, "ivf_offsets.bin");
        var bbmin = Path.Combine(dataDir, "ivf_bbox_min.bin");
        var bbmax = Path.Combine(dataDir, "ivf_bbox_max.bin");
        using var ds = Dataset.Open(vec, lab,
            File.Exists(q8) ? q8 : null,
            File.Exists(q8Soa) ? q8Soa : null,
            File.Exists(q16) ? q16 : null,
            null,
            cents,
            File.Exists(offs) ? offs : null,
            File.Exists(bbmin) ? bbmin : null,
            File.Exists(bbmax) ? bbmax : null,
            File.Exists(q16Blocked) ? q16Blocked : null,
            File.Exists(blockOffs) ? blockOffs : null);
        Console.WriteLine($"Loaded {ds.Count} refs.");

        // Build with possibly-overridden MinCount via reflection of the static field.
        // (The class lives in Api; we just call Build then re-build the table here.)
        ProfileFastPath.Build(ds);
        Console.WriteLine($"FastPath: used={ProfileFastPath.UsedBuckets} legit={ProfileFastPath.DecidedLegit} fraud={ProfileFastPath.DecidedFraud}");

        var fp2ConfigPath = Path.Combine(root, "resources/profile_fastpath2.json");
        ProfileFastPath2.Build(ds, fp2ConfigPath);
        if (ProfileFastPath2.IsEnabled)
            Console.WriteLine($"FastPath2: used={ProfileFastPath2.UsedBuckets} legit={ProfileFastPath2.DecidedLegit} fraud={ProfileFastPath2.DecidedFraud}");

        var bytes = File.ReadAllBytes(testData);
        using var doc = JsonDocument.Parse(bytes);
        var entries = doc.RootElement.GetProperty("entries");
        int total = entries.GetArrayLength();
        Console.WriteLine($"Auditing {total} entries...");

        var qBuf = new float[Dataset.Dimensions];
        int hits = 0, fp1Hits = 0, fp2Hits = 0, fpFastPath = 0, fnFastPath = 0;
        int fp1Errors = 0, fp2Errors = 0;
        int idx = 0;
        foreach (var entry in entries.EnumerateArray())
        {
            var req = entry.GetProperty("request");
            var raw = Encoding.UTF8.GetBytes(req.GetRawText());
            jvec.VectorizeJson(raw, qBuf);
            int expectedFc = (int)Math.Round(entry.GetProperty("expected_fraud_score").GetSingle() * 5f); // 0..5
            bool expectedFraud = expectedFc >= 3; // approved threshold = score < 0.6 ⇒ score ≥ 0.6 = fraud

            byte fp = ProfileFastPath.TryLookup(qBuf);
            bool fp2Hit = false;
            if (fp != ProfileFastPath.ResultUndecided)
            {
                fp1Hits++;
            }
            else if (ProfileFastPath2.IsEnabled)
            {
                fp = ProfileFastPath2.TryLookup(qBuf);
                fp2Hit = fp != ProfileFastPath2.ResultUndecided;
                if (fp2Hit) fp2Hits++;
            }
            if (fp == ProfileFastPath.ResultUndecided) { idx++; continue; }
            hits++;
            bool predFraud = fp == ProfileFastPath.ResultFraud;
            if (predFraud && !expectedFraud)
            {
                fpFastPath++;
                if (fp2Hit) fp2Errors++; else fp1Errors++;
                var id = req.GetProperty("id").GetString();
                Console.WriteLine($"FP @ idx={idx} id={id} source={(fp2Hit ? "fp2" : "fp1")} expected={expectedFc} pred=fraud(5)");
            }
            else if (!predFraud && expectedFraud)
            {
                fnFastPath++;
                if (fp2Hit) fp2Errors++; else fp1Errors++;
                var id = req.GetProperty("id").GetString();
                Console.WriteLine($"FN @ idx={idx} id={id} source={(fp2Hit ? "fp2" : "fp1")} expected={expectedFc} pred=legit(0)");
            }
            idx++;
        }

        Console.WriteLine($"--- summary (min_count={minCount}, computed by Build) ---");
        Console.WriteLine($"hits={hits}/{total} ({100.0*hits/total:F2}%)  fp1={fp1Hits}  fp2={fp2Hits}  FP={fpFastPath}  FN={fnFastPath}  fp1_errors={fp1Errors}  fp2_errors={fp2Errors}");
        return (fpFastPath + fnFastPath) > 0 ? 1 : 0;
    }
}
