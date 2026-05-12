using System.Globalization;
using System.Text;
using System.Text.Json;
using Rinha.Api;

namespace Rinha.Bench;

/// <summary>
/// Standalone audit: runs JsonVectorizer + SelectiveDecisionCascade against
/// test-data.json using the production selective-table order, reports FP/FN
/// attributable to the stages, and prints offending entries.
/// </summary>
public static class AuditFastPath
{
    public static int Run(string[] args)
    {
        string testData = "/tmp/rinha-eval/test/test-data.json";
        string? dataDirArg = null;
        string? configArg = null;
        int minCount = 100;
        foreach (var a in args)
        {
            if (a.StartsWith("--test-data=")) testData = a[12..];
            else if (a.StartsWith("--data-dir=")) dataDirArg = a[11..];
            else if (a.StartsWith("--selective-config=")) configArg = a[19..];
            else if (a.StartsWith("--min-count=")) minCount = int.Parse(a[12..], CultureInfo.InvariantCulture);
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

        var configPath = configArg ?? Path.Combine(root, "resources/selective_decision_tables.json");
        var cascade = SelectiveDecisionCascade.Build(ds, configPath);
        var stages = cascade.Stages;
        for (int i = 0; i < stages.Count; i++)
            Console.WriteLine($"Selective stage {i}: name={stages[i].Name} mode={stages[i].Mode} counts=[{string.Join(",", stages[i].DecidedByCount)}]");

        var bytes = File.ReadAllBytes(testData);
        using var doc = JsonDocument.Parse(bytes);
        var entries = doc.RootElement.GetProperty("entries");
        int total = entries.GetArrayLength();
        Console.WriteLine($"Auditing {total} entries...");

        var qBuf = new float[Dataset.Dimensions];
        int hits = 0, fpFastPath = 0, fnFastPath = 0;
        var stageHits = new int[stages.Count];
        var stageErrors = new int[stages.Count];
        var stageCountMismatch = new int[stages.Count];
        int idx = 0;
        foreach (var entry in entries.EnumerateArray())
        {
            var req = entry.GetProperty("request");
            var raw = Encoding.UTF8.GetBytes(req.GetRawText());
            jvec.VectorizeJson(raw, qBuf);
            int expectedFc = (int)Math.Round(entry.GetProperty("expected_fraud_score").GetSingle() * 5f); // 0..5
            bool expectedFraud = expectedFc >= 3; // approved threshold = score < 0.6 ⇒ score ≥ 0.6 = fraud

            byte pred = cascade.TryLookupWithStage(qBuf, out int stageIndex);
            if (pred == SelectiveDecisionCascade.ResultUndecided) { idx++; continue; }

            hits++;
            int predCount = pred;
            stageHits[stageIndex]++;
            if (predCount != expectedFc) stageCountMismatch[stageIndex]++;
            bool predFraud = predCount >= 3;
            if (predFraud && !expectedFraud)
            {
                fpFastPath++;
                stageErrors[stageIndex]++;
                var id = req.GetProperty("id").GetString();
                Console.WriteLine($"FP @ idx={idx} id={id} source={stages[stageIndex].Name} expected={expectedFc} pred={predCount}");
            }
            else if (!predFraud && expectedFraud)
            {
                fnFastPath++;
                stageErrors[stageIndex]++;
                var id = req.GetProperty("id").GetString();
                Console.WriteLine($"FN @ idx={idx} id={id} source={stages[stageIndex].Name} expected={expectedFc} pred={predCount}");
            }
            idx++;
        }

        Console.WriteLine($"--- summary (min_count={minCount}, computed by Build) ---");
        Console.WriteLine($"hits={hits}/{total} ({100.0*hits/total:F2}%)  FP={fpFastPath}  FN={fnFastPath}");
        for (int i = 0; i < stages.Count; i++)
            Console.WriteLine($"stage[{i}]={stages[i].Name} hits={stageHits[i]} errors={stageErrors[i]} count_mismatch={stageCountMismatch[i]}");
        return (fpFastPath + fnFastPath) > 0 ? 1 : 0;
    }
}
