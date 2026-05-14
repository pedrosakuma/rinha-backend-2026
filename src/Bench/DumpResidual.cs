using System.Globalization;
using System.Text;
using System.Text.Json;
using Rinha.Api;
using Rinha.Api.Scorers;

namespace Rinha.Bench;

/// <summary>
/// Per-query offline dump for borderline-residual analysis. For every query in
/// test-data.json emits: features, gt_5 (0..5), cascade_stage (-1 miss, else
/// stage index), and the IVF-resolved fraud_count (when stage&lt;0). Used by an
/// external analyzer to discover new fast-path stages that decide borderline
/// queries currently routed to IVF.
/// Usage: --dump-residual [--test-data=...] [--out=...] [--data-dir=...]
/// </summary>
public static class DumpResidual
{
    public static int Run(string[] args)
    {
        string testData = "/tmp/rinha-eval/test/test-data.json";
        string outPath = "/tmp/residual_dump.csv";
        string? dataDirArg = null;
        int nProbe = 1;
        foreach (var a in args)
        {
            if (a.StartsWith("--test-data=")) testData = a[12..];
            else if (a.StartsWith("--out=")) outPath = a[6..];
            else if (a.StartsWith("--data-dir=")) dataDirArg = a[11..];
            else if (a.StartsWith("--nprobe=")) nProbe = int.Parse(a[9..], CultureInfo.InvariantCulture);
        }

        string root = AppContext.BaseDirectory;
        while (root is not null && !File.Exists(Path.Combine(root, "Rinha.slnx")))
            root = Path.GetDirectoryName(root);
        if (root is null) throw new InvalidOperationException("repo root not found");
        var dataDir = dataDirArg ?? Path.Combine(root, "data");

        var norm = NormalizationConstants.Load(Path.Combine(root, "resources/normalization.json"));
        var mcc = MccRiskTable.Load(Path.Combine(root, "resources/mcc_risk.json"));
        var jvec = new JsonVectorizer(norm, mcc);

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
        var scorer = new IvfBlockedScorer(ds, nProbe);

        var bytes = File.ReadAllBytes(testData);
        using var doc = JsonDocument.Parse(bytes);
        var entries = doc.RootElement.GetProperty("entries");
        int total = entries.GetArrayLength();

        using var fs = File.Create(outPath);
        using var sw = new StreamWriter(fs);
        // Header: idx, gt_5, stage, ivf_count, f0..f13
        sw.Write("idx,gt_5,stage,ivf_count");
        for (int d = 0; d < Dataset.Dimensions; d++) sw.Write($",f{d}");
        sw.WriteLine();

        var qBuf = new float[Dataset.Dimensions];
        int idx = 0;
        int written = 0;
        foreach (var entry in entries.EnumerateArray())
        {
            var req = entry.GetProperty("request");
            var raw = Encoding.UTF8.GetBytes(req.GetRawText());
            jvec.VectorizeJson(raw, qBuf);
            float gt = entry.GetProperty("expected_fraud_score").GetSingle();
            int gt5 = Math.Clamp((int)Math.Round(gt * 5f), 0, 5);

            byte casc = cascade.TryLookupWithStage(qBuf, out int stage);
            int stageOut = casc == SelectiveDecisionCascade.ResultUndecided ? -1 : stage;
            int ivfCount = -1;
            if (stageOut < 0)
                ivfCount = scorer.ScoreCount(qBuf);

            sw.Write($"{idx},{gt5},{stageOut},{ivfCount}");
            for (int d = 0; d < Dataset.Dimensions; d++)
                sw.Write($",{qBuf[d].ToString("R", CultureInfo.InvariantCulture)}");
            sw.WriteLine();
            idx++;
            written++;
            if (written % 10000 == 0) Console.WriteLine($"  ... {written}/{total}");
        }
        Console.WriteLine($"Wrote {written} rows to {outPath}");
        return 0;
    }
}
