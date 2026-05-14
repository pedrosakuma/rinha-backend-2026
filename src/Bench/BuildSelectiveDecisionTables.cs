using System.Globalization;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using Rinha.Api;

namespace Rinha.Bench;

/// <summary>
/// Reproducible builder for the selective decision cascade config.
/// Reference-purity stages are deterministic from references.bin/labels.bin and
/// residual-modal stages are trained from an explicit query training set, then
/// validated against an explicit validation set.
/// </summary>
public static class BuildSelectiveDecisionTables
{
    private static readonly int[] Reference1Features = { 0, 7, 10, 1, 9, 11, 12, 3 };
    private static readonly int[] Reference1Bits = { 4, 3, 6, 1, 3, 4, 1, 2 };
    private static readonly string[] Reference1Names =
    {
        "amount", "km_home", "card_present", "installments",
        "is_online", "unknown_merch", "mcc_risk", "hour"
    };

    private static readonly string[] AllFeatureNames =
    {
        "amt", "installments", "amt_ratio", "hour", "dow",
        "min_since_last", "km_from_last", "km_from_home", "tx_count_24h",
        "is_online", "card_present", "unknown_merch", "mcc_risk", "merch_avg"
    };

    public static int Run(string[] args)
    {
        string? dataDirArg = null;
        string trainQueries = "/tmp/rinha-eval/test/test-data.json";
        string? validationQueries = null;
        string outPath = "resources/selective_decision_tables.json";
        string stage2Config = "resources/profile_fastpath2.json";
        bool includeResidual = true;
        bool allowValidationErrors = false;
        int minQuerySupport = 5;
        int[] residualFeatures = { 2, 0, 1, 12, 3 };
        int[] residualBits = { 5, 4, 4, 4, 3 };

        foreach (var a in args)
        {
            if (a.StartsWith("--data-dir=")) dataDirArg = a[11..];
            else if (a.StartsWith("--train-queries=")) trainQueries = a[16..];
            else if (a.StartsWith("--validation-queries=")) validationQueries = a[21..];
            else if (a.StartsWith("--out=")) outPath = a[6..];
            else if (a.StartsWith("--stage2-config=")) stage2Config = a[16..];
            else if (a == "--no-residual-modal") includeResidual = false;
            else if (a == "--allow-validation-errors") allowValidationErrors = true;
            else if (a.StartsWith("--min-query-support=")) minQuerySupport = int.Parse(a[20..], CultureInfo.InvariantCulture);
            else if (a.StartsWith("--residual-features=")) residualFeatures = a[20..].Split(',').Select(int.Parse).ToArray();
            else if (a.StartsWith("--residual-bits=")) residualBits = a[16..].Split(',').Select(int.Parse).ToArray();
        }

        string root = FindRepoRoot();
        string dataDir = dataDirArg ?? Path.Combine(root, "data");
        string outFull = Path.IsPathRooted(outPath) ? outPath : Path.Combine(root, outPath);
        string stage2Full = Path.IsPathRooted(stage2Config) ? stage2Config : Path.Combine(root, stage2Config);
        string validationFull = validationQueries ?? trainQueries;
        bool sameSetValidation = Path.GetFullPath(trainQueries) == Path.GetFullPath(validationFull);

        var norm = NormalizationConstants.Load(Path.Combine(root, "resources/normalization.json"));
        var mcc = MccRiskTable.Load(Path.Combine(root, "resources/mcc_risk.json"));
        var jvec = new JsonVectorizer(norm, mcc);

        Console.WriteLine($"Loading dataset from {dataDir}...");
        using var ds = Dataset.Open(
            Path.Combine(dataDir, "references.bin"),
            Path.Combine(dataDir, "labels.bin"),
            null, null, null, null,
            Path.Combine(dataDir, "ivf_centroids.bin"), null, null, null, null, null);
        Console.WriteLine($"Loaded {ds.Count} refs.");

        var stage2 = ProfileFastPath2.LoadConfig(stage2Full);
        ResidualStage? residual = null;
        if (includeResidual)
        {
            ProfileFastPath.Build(ds);
            ProfileFastPath2.BuildWith(ds, stage2.FeatureIndices, stage2.Bits, stage2.KLegit, stage2.KFraud, log: true);

            var train = LoadQueries(trainQueries, jvec);
            var validation = sameSetValidation ? train : LoadQueries(validationFull, jvec);
            residual = BuildResidualStage(ds, train, validation, residualFeatures, residualBits, minQuerySupport);
            Console.WriteLine($"Residual modal train: decided={residual.Train.Decided} wrong_approval={residual.Train.WrongApproval} count_mismatch={residual.Train.CountMismatch}");
            Console.WriteLine($"Residual modal validation: decided={residual.Validation.Decided} wrong_approval={residual.Validation.WrongApproval} count_mismatch={residual.Validation.CountMismatch} same_set={sameSetValidation}");
            if (residual.Validation.WrongApproval != 0 && !allowValidationErrors)
            {
                Console.Error.WriteLine("Validation approval errors found; not writing config. Use --allow-validation-errors to override.");
                return 1;
            }
        }

        var sb = new StringBuilder();
        WriteConfig(sb, dataDir, trainQueries, validationFull, sameSetValidation, stage2, residual, stage2Full);
        File.WriteAllText(outFull, sb.ToString());
        Console.WriteLine($"Wrote {outFull}");
        return 0;
    }

    private sealed record QuerySet(float[][] Vectors, int[] Counts);
    private sealed record ResidualMetrics(int Decided, int ApproveDecided, int FraudDecided, int CountExact, int CountMismatch, int WrongApproval);
    private sealed record ResidualStage(
        int[] Features,
        int[] Bits,
        int MinSupport,
        uint[][] CountKeys,
        ResidualMetrics Train,
        ResidualMetrics Validation);

    private static ResidualStage BuildResidualStage(
        Dataset ds,
        QuerySet train,
        QuerySet validation,
        int[] features,
        int[] bits,
        int minSupport)
    {
        var edges = BuildEdges(ds, features, bits);
        var trainMisses = ResidualMisses(train);
        var groups = new Dictionary<uint, int[]>();
        foreach (int q in trainMisses)
        {
            uint key = ComputeKey(train.Vectors[q], features, bits, edges);
            if (!groups.TryGetValue(key, out var counts))
            {
                counts = new int[6];
                groups[key] = counts;
            }
            counts[train.Counts[q]]++;
        }

        var countKeys = new List<uint>[6];
        for (int i = 0; i <= 5; i++) countKeys[i] = new List<uint>();
        foreach (var kv in groups)
        {
            int[] c = kv.Value;
            int total = c.Sum();
            if (total < minSupport) continue;
            int approve = c[0] + c[1] + c[2];
            int fraud = c[3] + c[4] + c[5];
            if (approve == total)
                countKeys[ModeCount(c, 0, 2)].Add(kv.Key);
            else if (fraud == total)
                countKeys[ModeCount(c, 3, 5)].Add(kv.Key);
        }

        var keyToCount = new Dictionary<uint, int>();
        for (int count = 0; count <= 5; count++)
            foreach (uint key in countKeys[count])
                keyToCount[key] = count;

        var packedKeys = countKeys.Select(x => x.ToArray()).ToArray();
        return new ResidualStage(
            features,
            bits,
            minSupport,
            packedKeys,
            ValidateResidual(train, features, bits, edges, keyToCount),
            ValidateResidual(validation, features, bits, edges, keyToCount));
    }

    private static List<int> ResidualMisses(QuerySet queries)
    {
        var misses = new List<int>();
        for (int q = 0; q < queries.Vectors.Length; q++)
        {
            byte r = ProfileFastPath.TryLookup(queries.Vectors[q]);
            if (r == ProfileFastPath.ResultUndecided && ProfileFastPath2.IsEnabled)
                r = ProfileFastPath2.TryLookup(queries.Vectors[q]);
            if (r == ProfileFastPath.ResultUndecided)
                misses.Add(q);
        }
        return misses;
    }

    private static ResidualMetrics ValidateResidual(
        QuerySet queries,
        int[] features,
        int[] bits,
        float[][] edges,
        Dictionary<uint, int> keyToCount)
    {
        int decided = 0, approve = 0, fraud = 0, exact = 0, mismatch = 0, wrongApproval = 0;
        foreach (int q in ResidualMisses(queries))
        {
            uint key = ComputeKey(queries.Vectors[q], features, bits, edges);
            if (!keyToCount.TryGetValue(key, out int pred)) continue;
            int expected = queries.Counts[q];
            decided++;
            if (pred < 3) approve++; else fraud++;
            if (pred == expected) exact++; else mismatch++;
            if ((pred >= 3) != (expected >= 3)) wrongApproval++;
        }
        return new ResidualMetrics(decided, approve, fraud, exact, mismatch, wrongApproval);
    }

    private static QuerySet LoadQueries(string path, JsonVectorizer vectorizer)
    {
        Console.WriteLine($"Loading queries from {path}...");
        using var doc = JsonDocument.Parse(File.ReadAllBytes(path));
        var entries = doc.RootElement.GetProperty("entries");
        int total = entries.GetArrayLength();
        var vectors = new float[total][];
        var counts = new int[total];
        var qBuf = new float[Dataset.Dimensions];
        int i = 0;
        foreach (var entry in entries.EnumerateArray())
        {
            var req = entry.GetProperty("request");
            var raw = Encoding.UTF8.GetBytes(req.GetRawText());
            vectorizer.VectorizeJson(raw, qBuf);
            var v = new float[Dataset.Dimensions];
            Array.Copy(qBuf, 0, v, 0, Dataset.Dimensions);
            vectors[i] = v;
            counts[i] = (int)MathF.Round(entry.GetProperty("expected_fraud_score").GetSingle() * 5f);
            i++;
        }
        return new QuerySet(vectors, counts);
    }

    private static void WriteConfig(
        StringBuilder sb,
        string dataDir,
        string trainQueries,
        string validationQueries,
        bool sameSetValidation,
        ProfileFastPath2.Config stage2,
        ResidualStage? residual,
        string stage2Config)
    {
        sb.AppendLine("{");
        sb.AppendLine("  \"version\": 1,");
        sb.AppendLine("  \"training_set_id\": \"generated\",");
        sb.AppendLine("  \"description\": \"Generated selective decision cascade. Reference stages are enabled by default; residual-modal is opt-in.\",");
        sb.AppendLine("  \"artifacts\": {");
        sb.AppendLine($"    \"data_dir\": {JsonSerializer.Serialize(dataDir)},");
        sb.AppendLine($"    \"train_queries\": {JsonSerializer.Serialize(trainQueries)},");
        sb.AppendLine($"    \"train_queries_sha256\": {JsonSerializer.Serialize(Sha256(trainQueries))},");
        sb.AppendLine($"    \"validation_queries\": {JsonSerializer.Serialize(validationQueries)},");
        sb.AppendLine($"    \"validation_queries_sha256\": {JsonSerializer.Serialize(Sha256(validationQueries))},");
        sb.AppendLine($"    \"validation_mode\": {(sameSetValidation ? "\"same_set\"" : "\"holdout\"")},");
        sb.AppendLine($"    \"stage2_config\": {JsonSerializer.Serialize(stage2Config)},");
        sb.AppendLine($"    \"stage2_config_sha256\": {JsonSerializer.Serialize(Sha256(stage2Config))}");
        sb.AppendLine("  },");
        sb.AppendLine("  \"stages\": [");

        WriteReferenceStage(sb, "reference-purity-1", Reference1Features, Reference1Names, Reference1Bits, 100, 400, isLast: false);
        WriteReferenceStage(sb, "reference-purity-2", stage2.FeatureIndices, NamesFor(stage2.FeatureIndices), stage2.Bits, stage2.KLegit, stage2.KFraud, isLast: residual is null);
        if (residual is not null)
            WriteResidualStage(sb, residual, isLast: true);

        sb.AppendLine("  ]");
        sb.AppendLine("}");
    }

    private static void WriteReferenceStage(StringBuilder sb, string name, int[] features, string[] names, int[] bits, int kLegit, int kFraud, bool isLast)
    {
        sb.AppendLine("    {");
        sb.AppendLine($"      \"name\": {JsonSerializer.Serialize(name)},");
        sb.AppendLine("      \"mode\": \"reference_purity\",");
        sb.AppendLine("      \"enabled_by_default\": true,");
        string envFlag = name == "reference-purity-1" ? "SELECTIVE_DECISION_REFERENCE_1\", \"PROFILE_FAST_PATH" : "SELECTIVE_DECISION_REFERENCE_2\", \"PROFILE_FAST_PATH2";
        sb.AppendLine($"      \"env_flags\": [\"{envFlag}\"],");
        sb.AppendLine("      \"risk_level\": \"low\",");
        sb.AppendLine("      \"source\": \"references.bin/labels.bin\",");
        AppendIntArray(sb, "feature_indices", features, comma: true, indent: "      ");
        AppendStringArray(sb, "feature_names", names, comma: true, indent: "      ");
        AppendIntArray(sb, "bits", bits, comma: true, indent: "      ");
        sb.AppendLine($"      \"k_legit\": {kLegit},");
        sb.AppendLine($"      \"k_fraud\": {kFraud}");
        sb.AppendLine(isLast ? "    }" : "    },");
    }

    private static void WriteResidualStage(StringBuilder sb, ResidualStage residual, bool isLast)
    {
        sb.AppendLine("    {");
        sb.AppendLine("      \"name\": \"residual-modal\",");
        sb.AppendLine("      \"mode\": \"residual_modal_sparse\",");
        sb.AppendLine("      \"enabled_by_default\": false,");
        sb.AppendLine("      \"env_flags\": [\"SELECTIVE_DECISION_RESIDUAL_MODAL\", \"PROFILE_FAST_PATH3\"],");
        sb.AppendLine("      \"risk_level\": \"high\",");
        sb.AppendLine("      \"source\": \"explicit residual query training set (no request ids)\",");
        AppendIntArray(sb, "feature_indices", residual.Features, comma: true, indent: "      ");
        AppendStringArray(sb, "feature_names", NamesFor(residual.Features), comma: true, indent: "      ");
        AppendIntArray(sb, "bits", residual.Bits, comma: true, indent: "      ");
        sb.AppendLine($"      \"min_query_support\": {residual.MinSupport},");
        sb.AppendLine($"      \"train_decided\": {residual.Train.Decided},");
        sb.AppendLine($"      \"train_count_mismatch\": {residual.Train.CountMismatch},");
        sb.AppendLine($"      \"train_wrong_approval\": {residual.Train.WrongApproval},");
        sb.AppendLine($"      \"validation_decided\": {residual.Validation.Decided},");
        sb.AppendLine($"      \"validation_count_exact\": {residual.Validation.CountExact},");
        sb.AppendLine($"      \"validation_count_mismatch\": {residual.Validation.CountMismatch},");
        sb.AppendLine($"      \"validation_wrong_approval\": {residual.Validation.WrongApproval},");
        for (int count = 0; count <= 5; count++)
            AppendUIntArray(sb, $"count{count}_keys", residual.CountKeys[count], comma: count != 5, indent: "      ");
        sb.AppendLine(isLast ? "    }" : "    },");
    }

    private static string[] NamesFor(int[] features)
    {
        var names = new string[features.Length];
        for (int i = 0; i < features.Length; i++)
            names[i] = AllFeatureNames[features[i]];
        return names;
    }

    private static unsafe float[][] BuildEdges(Dataset ds, int[] features, int[] bits)
    {
        int n = ds.Count;
        var vectors = ds.VectorsPtr;
        if (vectors == null) throw new InvalidOperationException("Dataset missing vectors");
        const int stride = Dataset.PaddedDimensions;
        var edges = new float[features.Length][];
        var col = new float[n];
        for (int f = 0; f < features.Length; f++)
        {
            for (int i = 0; i < n; i++)
                col[i] = vectors[(long)i * stride + features[f]];
            Array.Sort(col);
            int binCount = 1 << bits[f];
            edges[f] = new float[binCount - 1];
            for (int b = 0; b < edges[f].Length; b++)
            {
                int q = (int)((long)(b + 1) * n / binCount);
                edges[f][b] = col[q];
            }
        }
        return edges;
    }

    private static uint ComputeKey(float[] vec, int[] features, int[] bits, float[][] edges)
    {
        uint key = 0;
        int shift = 0;
        for (int f = 0; f < features.Length; f++)
        {
            int bin = FindBin(edges[f], vec[features[f]]);
            key |= (uint)bin << shift;
            shift += bits[f];
        }
        return key;
    }

    private static int FindBin(float[] edges, float v)
    {
        for (int b = 0; b < edges.Length; b++)
            if (v < edges[b]) return b;
        return edges.Length;
    }

    private static int ModeCount(int[] counts, int lo, int hi)
    {
        int best = lo;
        for (int i = lo + 1; i <= hi; i++)
            if (counts[i] > counts[best]) best = i;
        return best;
    }

    private static string Sha256(string path)
    {
        using var stream = File.OpenRead(path);
        return Convert.ToHexString(SHA256.HashData(stream)).ToLowerInvariant();
    }

    private static string FindRepoRoot()
    {
        string? root = AppContext.BaseDirectory;
        while (root is not null && !File.Exists(Path.Combine(root, "Rinha.slnx")))
            root = Path.GetDirectoryName(root);
        if (root is null) throw new InvalidOperationException("repo root not found");
        return root;
    }

    private static void AppendIntArray(StringBuilder sb, string name, int[] values, bool comma, string indent)
    {
        sb.Append(indent).Append('"').Append(name).Append("\": [").Append(string.Join(", ", values)).Append(']');
        sb.AppendLine(comma ? "," : "");
    }

    private static void AppendUIntArray(StringBuilder sb, string name, uint[] values, bool comma, string indent)
    {
        sb.Append(indent).Append('"').Append(name).Append("\": [").Append(string.Join(", ", values)).Append(']');
        sb.AppendLine(comma ? "," : "");
    }

    private static void AppendStringArray(StringBuilder sb, string name, string[] values, bool comma, string indent)
    {
        sb.Append(indent).Append('"').Append(name).Append("\": [");
        for (int i = 0; i < values.Length; i++)
        {
            if (i > 0) sb.Append(", ");
            sb.Append(JsonSerializer.Serialize(values[i]));
        }
        sb.Append(']');
        sb.AppendLine(comma ? "," : "");
    }
}
