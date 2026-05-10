using System.Globalization;
using System.Text;
using System.Text.Json;
using Rinha.Api;

namespace Rinha.Bench;

/// <summary>
/// Standalone helper: vectorizes every entry from test-data.json and writes
/// a flat binary file: 54100 rows × (14 float32 features + 1 byte
/// expected_fraud_score). Used by /tmp/analyze_eval_buckets.py.
/// </summary>
public static class DumpFeatures
{
    public static int Run(string[] args)
    {
        string testData = "/tmp/rinha-eval/test/test-data.json";
        string outPath = "/tmp/eval_features.bin";
        int limit = int.MaxValue;
        foreach (var a in args)
        {
            if (a.StartsWith("--test-data=")) testData = a[12..];
            else if (a.StartsWith("--out=")) outPath = a[6..];
            else if (a.StartsWith("--limit=")) limit = int.Parse(a[8..], CultureInfo.InvariantCulture);
        }

        // Inline repo root finder (Replay.FindRepoRoot is private).
        string root = AppContext.BaseDirectory;
        while (root is not null && !File.Exists(Path.Combine(root, "Rinha.slnx")))
            root = Path.GetDirectoryName(root);
        if (root is null) throw new InvalidOperationException("repo root not found");
        var norm = NormalizationConstants.Load(Path.Combine(root, "resources/normalization.json"));
        var mcc = MccRiskTable.Load(Path.Combine(root, "resources/mcc_risk.json"));
        var jvec = new JsonVectorizer(norm, mcc);
        var bytes = File.ReadAllBytes(testData);
        using var doc = JsonDocument.Parse(bytes);
        var entries = doc.RootElement.GetProperty("entries");
        int total = Math.Min(entries.GetArrayLength(), limit);
        Console.WriteLine($"Dumping {total} entries from {testData} -> {outPath}");

        using var fs = File.Create(outPath);
        using var bw = new BinaryWriter(fs);
        var qBuf = new float[Dataset.Dimensions];
        int n = 0;
        foreach (var entry in entries.EnumerateArray())
        {
            if (n >= limit) break;
            var req = entry.GetProperty("request");
            var raw = Encoding.UTF8.GetBytes(req.GetRawText());
            jvec.VectorizeJson(raw, qBuf);
            for (int d = 0; d < Dataset.Dimensions; d++)
                bw.Write(qBuf[d]);
            float score = entry.TryGetProperty("expected_fraud_score", out var sc) ? sc.GetSingle() : 0f;
            // expected_fraud_score is in {0..5}; write as int8 fraud_count
            bw.Write((byte)Math.Clamp((int)Math.Round(score * 5f), 0, 5));
            n++;
        }
        Console.WriteLine($"Wrote {n} rows; size={fs.Length} bytes");
        return 0;
    }
}
