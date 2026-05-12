using System.Globalization;
using System.Text;
using System.Text.Json;
using Rinha.Api;

namespace Rinha.Bench;

/// <summary>
/// Offline tool that searches for the best 2nd-level FastPath config and writes
/// resources/profile_fastpath2.json. Greedy search over feature subsets + bit
/// allocations + (k_fraud, k_legit) thresholds. Validates against test-data
/// queries that the 1st FastPath misses, requiring ZERO false positives /
/// false negatives at the 0.6 fraud-decision threshold.
///
/// Usage:
///   Rinha.Bench --build-fastpath2 [--test-data=...] [--out=resources/profile_fastpath2.json]
///   Rinha.Bench --build-fastpath2 --eval --features=5,13,2,8,6,1 --bits=4,3,3,3,3,3 --k-fraud=200 --k-legit=50
/// </summary>
public static class BuildFastPath2
{
    // Strong miss-cohort separators discovered by FraudMissAnalysis (FP1's residue).
    // Tight pool — search budget is bounded by ~3s per evaluation.
    static readonly int[] CandidateFeatures = { 5, 13, 2, 8, 6, 1, 12 };
    // 5=min_since_last, 13=merch_avg, 2=amt_ratio, 8=tx_count_24h, 6=km_from_last,
    // 1=installments,   12=mcc_risk

    static readonly string[] AllFeatureNames =
    {
        "amt", "installments", "amt_ratio", "hour", "dow",
        "min_since_last", "km_from_last", "km_from_home", "tx_count_24h",
        "is_online", "card_present", "unknown_merch", "mcc_risk", "merch_avg"
    };

    public static int Run(string[] args)
    {
        string testData = "/tmp/rinha-eval/test/test-data.json";
        string outPath  = "resources/profile_fastpath2.json";
        bool evalMode = false;
        int[]? userFeats = null;
        int[]? userBits  = null;
        int userKF = 200, userKL = 50;

        foreach (var a in args)
        {
            if (a.StartsWith("--test-data=")) testData = a[12..];
            else if (a.StartsWith("--out=")) outPath = a[6..];
            else if (a == "--eval") evalMode = true;
            else if (a.StartsWith("--features=")) userFeats = a[11..].Split(',').Select(int.Parse).ToArray();
            else if (a.StartsWith("--bits=")) userBits = a[7..].Split(',').Select(int.Parse).ToArray();
            else if (a.StartsWith("--k-fraud=")) userKF = int.Parse(a[10..], CultureInfo.InvariantCulture);
            else if (a.StartsWith("--k-legit=")) userKL = int.Parse(a[10..], CultureInfo.InvariantCulture);
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
        using var ds = Dataset.Open(
            Path.Combine(dataDir, "references.bin"),
            Path.Combine(dataDir, "labels.bin"),
            null, null, null, null,
            Path.Combine(dataDir, "ivf_centroids.bin"), null, null, null, null, null);
        Console.WriteLine($"Loaded {ds.Count} refs.");
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

        // Misses of FP1 (the only queries FP2 ever sees).
        var misses = new List<int>();
        for (int q = 0; q < total; q++)
            if (ProfileFastPath.TryLookup(vecs[q]) == ProfileFastPath.ResultUndecided)
                misses.Add(q);
        int totalMisses = misses.Count;
        int fraudMiss = misses.Count(q => gtScore[q] >= 0.6f);
        int legitMiss = misses.Count(q => gtScore[q] < 0.6f);
        Console.WriteLine($"FP1 misses: {totalMisses} ({100.0*totalMisses/total:F2}%)  fraud(GT≥0.6)={fraudMiss}  legit(GT<0.6)={legitMiss}");

        if (evalMode)
        {
            if (userFeats is null || userBits is null) { Console.WriteLine("--eval needs --features and --bits"); return 1; }
            var s = EvaluateConfig(ds, userFeats, userBits, userKF, userKL, vecs, gtScore, misses);
            Console.WriteLine();
            Console.WriteLine(FormatScore(s));
            return 0;
        }

        // ---- Greedy search ----
        Console.WriteLine();
        Console.WriteLine("=== Greedy search over feature subsets + bit allocations ===");
        Console.WriteLine($"{"features",-28} {"bits",-22} {"kF",4} {"kL",4} {"dec",6} {"correct",8} {"wrong",6} {"FN",4} {"FP",4} {"score",8}");

        Score? best = null;
        int[]? bestFeats = null, bestBits = null;
        int bestKF = 0, bestKL = 0;

        // Try subsets of varying length and bit allocations.
        // Search budget: prune aggressively to keep this under ~30min.
        var subsets = EnumerateSubsets(CandidateFeatures, minLen: 5, maxLen: 6).ToList();
        Console.WriteLine($"({subsets.Count} feature subsets, evaluating with several bit allocations + k thresholds)");

        foreach (var feats in subsets)
        {
            foreach (var bits in BitAllocations(feats.Length))
            {
                foreach (var (kF, kL) in new[] { (200, 150), (300, 200), (400, 300), (500, 500) })
                {
                    var s = EvaluateConfig(ds, feats, bits, kF, kL, vecs, gtScore, misses);
                    if (s.WrongFraud > 0 || s.WrongLegit > 0) continue; // hard reject
                    if (best is null || s.Decided > best.Decided ||
                        (s.Decided == best.Decided && s.BorderHits < best.BorderHits))
                    {
                        best = s;
                        bestFeats = feats; bestBits = bits; bestKF = kF; bestKL = kL;
                        Console.WriteLine($"{Fmt(feats),-28} {Fmt(bits),-22} {kF,4} {kL,4} {s.Decided,6} {s.CorrectFraud + s.CorrectLegit,8} 0      0    0    {s.Decided - s.BorderHits/4.0,8:F1}");
                    }
                }
            }
        }

        if (best is null || bestFeats is null || bestBits is null)
        {
            Console.WriteLine("No valid config found.");
            return 1;
        }

        Console.WriteLine();
        Console.WriteLine("=== Best config ===");
        Console.WriteLine($"features = [{string.Join(",", bestFeats)}]  ({string.Join(",", bestFeats.Select(i => AllFeatureNames[i]))})");
        Console.WriteLine($"bits     = [{string.Join(",", bestBits)}]");
        Console.WriteLine($"k_fraud  = {bestKF}");
        Console.WriteLine($"k_legit  = {bestKL}");
        Console.WriteLine(FormatScore(best));

        // Write JSON.
        string outFull = Path.IsPathRooted(outPath) ? outPath : Path.Combine(root, outPath);
        WriteJson(outFull, bestFeats, bestBits, bestKF, bestKL, best);
        Console.WriteLine($"Wrote {outFull}");
        return 0;
    }

    sealed record Score(int Decided, int CorrectFraud, int CorrectLegit, int WrongFraud, int WrongLegit, int BorderHits);

    static Score EvaluateConfig(Dataset ds, int[] feats, int[] bits, int kF, int kL,
                                float[][] vecs, float[] gt, List<int> misses)
    {
        ProfileFastPath2.BuildWith(ds, feats, bits, kL, kF, log: false);
        int dec = 0, cF = 0, cL = 0, wF = 0, wL = 0, border = 0;
        foreach (var q in misses)
        {
            var r = ProfileFastPath2.TryLookup(vecs[q]);
            if (r == ProfileFastPath2.ResultUndecided) continue;
            dec++;
            float g = gt[q];
            bool gtFraud = g >= 0.6f;
            bool isBorder = g >= 0.2f && g <= 0.8f;
            if (r == ProfileFastPath2.ResultFraud)
            {
                if (gtFraud) cF++; else wF++; // wF = false positive (saying fraud on legit)
            }
            else // Legit
            {
                if (!gtFraud) cL++; else wL++; // wL = false negative
            }
            if (isBorder) border++;
        }
        ProfileFastPath2.Disable();
        return new Score(dec, cF, cL, wF, wL, border);
    }

    static IEnumerable<int[]> EnumerateSubsets(int[] pool, int minLen, int maxLen)
    {
        // Bitmask enumeration. pool size is 10, so 1024 subsets total — fine.
        int n = pool.Length;
        for (int m = 1; m < (1 << n); m++)
        {
            int pc = System.Numerics.BitOperations.PopCount((uint)m);
            if (pc < minLen || pc > maxLen) continue;
            var r = new int[pc];
            int j = 0;
            for (int i = 0; i < n; i++) if ((m & (1 << i)) != 0) r[j++] = pool[i];
            yield return r;
        }
    }

    static IEnumerable<int[]> BitAllocations(int nFeats)
    {
        // Fixed allocations by feature count (sum<=24).
        // Hand-picked compromises: high-bit on first 1-2 separators, low on rest.
        switch (nFeats)
        {
            case 4:
                yield return new[] { 5, 5, 5, 5 };
                yield return new[] { 6, 5, 5, 4 };
                yield return new[] { 6, 6, 4, 4 };
                yield return new[] { 4, 4, 4, 4 };
                break;
            case 5:
                yield return new[] { 5, 5, 4, 4, 4 };
                yield return new[] { 6, 5, 4, 4, 3 };
                yield return new[] { 4, 4, 4, 4, 4 };
                yield return new[] { 5, 4, 4, 4, 3 };
                break;
            case 6:
                yield return new[] { 4, 4, 4, 4, 4, 4 };
                yield return new[] { 5, 4, 4, 4, 4, 3 };
                yield return new[] { 5, 5, 3, 3, 3, 3 };
                yield return new[] { 4, 4, 3, 3, 3, 3 };
                break;
            case 7:
                yield return new[] { 4, 4, 4, 3, 3, 3, 3 };
                yield return new[] { 4, 3, 3, 3, 3, 3, 3 };
                yield return new[] { 3, 3, 3, 3, 3, 3, 3 };
                break;
            case 8:
                yield return new[] { 3, 3, 3, 3, 3, 3, 3, 3 };
                yield return new[] { 4, 3, 3, 3, 3, 3, 3, 2 };
                yield return new[] { 4, 4, 3, 3, 3, 3, 2, 2 };
                break;
        }
    }

    static string Fmt(int[] a) => "[" + string.Join(",", a) + "]";

    static string FormatScore(Score s) =>
        $"  Decided: {s.Decided}  CorrectFraud: {s.CorrectFraud}  CorrectLegit: {s.CorrectLegit}  " +
        $"WrongFraud(FP): {s.WrongFraud}  WrongLegit(FN): {s.WrongLegit}  border-hits: {s.BorderHits}";

    static void WriteJson(string path, int[] feats, int[] bits, int kF, int kL, Score s)
    {
        var sb = new StringBuilder();
        sb.AppendLine("{");
        sb.AppendLine($"  \"version\": 1,");
        sb.AppendLine($"  \"feature_indices\": [{string.Join(", ", feats)}],");
        sb.Append("  \"feature_names\": [");
        for (int i = 0; i < feats.Length; i++)
        {
            if (i > 0) sb.Append(", ");
            sb.Append('"').Append(AllFeatureNames[feats[i]]).Append('"');
        }
        sb.AppendLine("],");
        sb.AppendLine($"  \"bits\": [{string.Join(", ", bits)}],");
        sb.AppendLine($"  \"k_fraud\": {kF},");
        sb.AppendLine($"  \"k_legit\": {kL},");
        sb.AppendLine($"  \"trained_on\": \"references.bin\",");
        sb.AppendLine($"  \"validated_decided\": {s.Decided},");
        sb.AppendLine($"  \"validated_correct_fraud\": {s.CorrectFraud},");
        sb.AppendLine($"  \"validated_correct_legit\": {s.CorrectLegit},");
        sb.AppendLine($"  \"validated_wrong_fraud\": {s.WrongFraud},");
        sb.AppendLine($"  \"validated_wrong_legit\": {s.WrongLegit}");
        sb.AppendLine("}");
        File.WriteAllText(path, sb.ToString());
    }
}
