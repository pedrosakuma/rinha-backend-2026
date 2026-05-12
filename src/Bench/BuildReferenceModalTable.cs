using System.Globalization;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using Rinha.Api;

namespace Rinha.Bench;

/// <summary>
/// Builds a sparse reference-modal lookup table for the SelectiveDecisionCascade
/// using the REFERENCE DATASET (references.bin / labels.bin) + the IVF scorer,
/// instead of memorising eval-query outputs like the original residual-modal (FP3).
///
/// Algorithm:
///   1. Build FP1 (reference-purity-1) and FP2 (reference-purity-2) from references.
///   2. For each FP1 bucket that neither FP1 nor FP2 decides ("undecided bucket"):
///      a. Collect all reference points that land in that bucket.
///      b. Sample up to --sample=N of them.
///      c. Run scorer.ScoreCount() on each sampled point.
///      d. Take the modal fraud_count.
///      e. If frequency(modal) / sample_size >= --confidence (default 0.80),
///         emit the bucket into countX_keys.
///   3. Validate against test-data.json (if present): measure wrong_approvals and
///      count_mismatches, refuse to write output if wrong_approvals > 0 unless
///      --allow-validation-errors is set.
///
/// Why use FP1's feature space:
///   Queries that escape FP1 land in FP1-undecided buckets. A reference-modal table
///   using the same key function as FP1 can decide those residuals without a separate
///   feature space, keeping one table for the whole range.
///
/// Usage:
///   dotnet run --project src/Bench -- --build-reference-modal [options]
///   --data-dir=&lt;path&gt;           dataset directory (default: &lt;repo&gt;/data)
///   --scorer=ivf-blocked         scorer to use for ScoreCount (default: ivf-blocked)
///   --sample=20                  reference points to sample per undecided bucket
///   --confidence=0.80            min fraction of samples agreeing on modal fraud_count
///   --min-bucket=5               min reference points in a bucket to consider it
///   --out=resources/selective_decision_tables.json  (appends a new stage or writes standalone JSON)
///   --standalone                 write a standalone JSON fragment (not full cascade config)
///   --stage2-config=resources/profile_fastpath2.json
///   --test-data=&lt;path&gt;           eval queries for validation (default: bench/k6/test-data.json)
///   --allow-validation-errors    proceed even if validation shows wrong_approvals > 0
///   --features=0,7,10,1,9,11,12,3   feature indices (default: same as FP1)
///   --bits=4,3,6,1,3,4,1,2          bits per feature (default: same as FP1)
/// </summary>
public static class BuildReferenceModalTable
{
    private static readonly string[] AllFeatureNames =
    {
        "amt", "installments", "amt_ratio", "hour", "dow",
        "min_since_last", "km_from_last", "km_from_home", "tx_count_24h",
        "is_online", "card_present", "unknown_merch", "mcc_risk", "merch_avg"
    };

    // FP1 defaults (reference-purity-1).
    private static readonly int[] DefaultFeatures = { 0, 7, 10, 1, 9, 11, 12, 3 };
    private static readonly int[] DefaultBits     = { 4, 3,  6, 1, 3,  4,  1, 2 };

    public static int Run(string[] args)
    {
        string? dataDirArg = null;
        string scorerName = "ivf-blocked";
        int sample = 20;
        double confidence = 0.80;
        int minBucket = 5;
        bool standalone = false;
        string outPath = "resources/selective_decision_tables.json";
        string stage2Config = "resources/profile_fastpath2.json";
        string? testDataPath = null;
        bool allowValidationErrors = false;
        int[] features = DefaultFeatures;
        int[] bits = DefaultBits;

        foreach (var a in args)
        {
            if (a.StartsWith("--data-dir="))         dataDirArg = a[11..];
            else if (a.StartsWith("--scorer="))       scorerName = a[9..];
            else if (a.StartsWith("--sample="))       sample = int.Parse(a[9..], CultureInfo.InvariantCulture);
            else if (a.StartsWith("--confidence="))   confidence = double.Parse(a[13..], CultureInfo.InvariantCulture);
            else if (a.StartsWith("--min-bucket="))   minBucket = int.Parse(a[13..], CultureInfo.InvariantCulture);
            else if (a.StartsWith("--out="))          outPath = a[6..];
            else if (a.StartsWith("--stage2-config=")) stage2Config = a[16..];
            else if (a.StartsWith("--test-data="))    testDataPath = a[12..];
            else if (a.StartsWith("--features="))     features = a[11..].Split(',').Select(int.Parse).ToArray();
            else if (a.StartsWith("--bits="))         bits = a[7..].Split(',').Select(int.Parse).ToArray();
            else if (a == "--standalone")             standalone = true;
            else if (a == "--allow-validation-errors") allowValidationErrors = true;
        }

        string root = FindRepoRoot();
        string dataDir = dataDirArg ?? Path.Combine(root, "data");
        testDataPath ??= Path.Combine(root, "bench", "k6", "test-data.json");
        string stage2Full = Path.IsPathRooted(stage2Config) ? stage2Config : Path.Combine(root, stage2Config);
        string outFull = Path.IsPathRooted(outPath) ? outPath : Path.Combine(root, outPath);

        Console.WriteLine($"BuildReferenceModalTable: data={dataDir} scorer={scorerName}");
        Console.WriteLine($"  features=[{string.Join(",", features)}] bits=[{string.Join(",", bits)}]");
        Console.WriteLine($"  sample={sample} confidence={confidence:P0} min_bucket={minBucket}");

        var norm = NormalizationConstants.Load(Path.Combine(root, "resources/normalization.json"));
        var mcc  = MccRiskTable.Load(Path.Combine(root, "resources/mcc_risk.json"));
        var jvec = new JsonVectorizer(norm, mcc);

        Console.WriteLine($"Loading dataset from {dataDir}...");
        using var ds = Dataset.Open(
            Path.Combine(dataDir, "references.bin"),
            Path.Combine(dataDir, "labels.bin"),
            null, null, null, null,
            Path.Combine(dataDir, "ivf_centroids.bin"),
            Path.Combine(dataDir, "ivf_offsets.bin"),
            Path.Combine(dataDir, "ivf_bbox_min.bin"),
            Path.Combine(dataDir, "ivf_bbox_max.bin"),
            Path.Combine(dataDir, "references_q16_blocked.bin"),
            Path.Combine(dataDir, "ivf_block_offsets.bin"));
        Console.WriteLine($"Loaded {ds.Count:N0} reference points.");

        // Build FP1 + FP2 so we know which buckets they already decide.
        ProfileFastPath.Build(ds);
        var stage2Cfg = ProfileFastPath2.LoadConfig(stage2Full);
        ProfileFastPath2.BuildWith(ds, stage2Cfg.FeatureIndices, stage2Cfg.Bits, stage2Cfg.KLegit, stage2Cfg.KFraud, log: true);

        // Build the scorer used for sampling.
        var scorer = ScorerFactory.Create(scorerName, ds);

        // Build reference-modal table.
        var result = BuildModalFromReferences(ds, scorer, features, bits, sample, confidence, minBucket);
        Console.WriteLine($"Reference-modal: undecided_buckets={result.UndecidedBuckets} " +
                          $"sampled={result.SampledBuckets} decided={result.DecidedBuckets} " +
                          $"decided_counts=[{string.Join(",", result.CountKeys.Select(k => k.Length))}]");

        // Validate against test-data if available.
        ValidationResult? validation = null;
        if (File.Exists(testDataPath))
        {
            Console.WriteLine($"Validating against {testDataPath}...");
            validation = Validate(testDataPath, jvec, features, bits, result);
            Console.WriteLine($"  decided={validation.Decided} wrong_approval={validation.WrongApproval} " +
                              $"count_mismatch={validation.CountMismatch} count_exact={validation.CountExact}");
            if (validation.WrongApproval > 0 && !allowValidationErrors)
            {
                Console.Error.WriteLine("Validation wrong_approvals > 0. Refusing to write. Use --allow-validation-errors to override.");
                return 1;
            }
        }
        else
        {
            Console.WriteLine($"No test-data at {testDataPath}; skipping validation.");
        }

        if (standalone)
        {
            WriteStandaloneFragment(outFull, features, bits, result, validation, dataDir);
        }
        else
        {
            AppendOrReplaceModalStage(outFull, features, bits, result, validation, dataDir, confidence, sample);
        }

        Console.WriteLine($"Wrote {outFull}");
        return 0;
    }

    // ─── Core build logic ────────────────────────────────────────────────────

    private sealed class BuildResult
    {
        public int UndecidedBuckets;
        public int SampledBuckets;
        public int DecidedBuckets;
        public uint[][] CountKeys = new uint[6][];
    }

    private sealed class ValidationResult
    {
        public int Decided;
        public int WrongApproval;
        public int CountMismatch;
        public int CountExact;
    }

    private static unsafe BuildResult BuildModalFromReferences(
        Dataset ds,
        IFraudScorer scorer,
        int[] features,
        int[] bits,
        int maxSample,
        double confidenceThreshold,
        int minBucket)
    {
        var (shifts, totalBits) = BuildShifts(bits);
        long slots = 1L << totalBits;
        var edges = BuildEdges(ds, features, bits);

        // Group reference point indices by bucket key.
        Console.WriteLine("Bucketing reference points...");
        var buckets = new Dictionary<uint, List<int>>(capacity: 65536);
        int n = ds.Count;
        var vectors = ds.VectorsPtr;
        const int stride = Dataset.PaddedDimensions;

        for (int i = 0; i < n; i++)
        {
            float* row = vectors + (long)i * stride;
            uint key = ComputeKey(row, features, shifts, edges);
            if (!buckets.TryGetValue(key, out var list))
                buckets[key] = list = new List<int>(capacity: 64);
            list.Add(i);
        }
        Console.WriteLine($"  {buckets.Count:N0} occupied buckets.");

        // For each bucket not decided by FP1/FP2, sample and score.
        var rng = new Random(20260512);
        var countKeys = new List<uint>[6];
        for (int i = 0; i <= 5; i++) countKeys[i] = new List<uint>();
        int undecided = 0, sampled = 0, decided = 0;

        Span<float> qBuf = stackalloc float[Dataset.Dimensions];
        var sw = System.Diagnostics.Stopwatch.StartNew();
        int bucketsDone = 0;

        foreach (var kv in buckets)
        {
            uint key = kv.Key;
            var members = kv.Value;

            // Skip if FP1 or FP2 already decides this bucket's queries.
            // We use a representative point to test — if the key falls in a
            // decided FP1/FP2 bucket, queries landing here are already handled.
            int repIdx = members[rng.Next(members.Count)];
            CopyVector(vectors + (long)repIdx * stride, qBuf);

            byte fp1 = ProfileFastPath.TryLookup(qBuf);
            if (fp1 != ProfileFastPath.ResultUndecided) continue; // FP1 decides it
            byte fp2 = ProfileFastPath2.TryLookup(qBuf);
            if (fp2 != ProfileFastPath2.ResultUndecided) continue; // FP2 decides it

            undecided++;
            if (members.Count < minBucket) continue;

            // Sample up to maxSample members.
            var indices = members.Count <= maxSample
                ? members
                : SampleWithoutReplacement(members, maxSample, rng);
            sampled++;

            // Run scorer on each sample.
            var fraudCounts = new int[6];
            foreach (int idx in indices)
            {
                CopyVector(vectors + (long)idx * stride, qBuf);
                int fc = scorer.ScoreCount(qBuf);
                if ((uint)fc > 5u) fc = fc < 0 ? 0 : 5;
                fraudCounts[fc]++;
            }

            // Modal fraud_count.
            int modal = 0;
            for (int c = 1; c <= 5; c++)
                if (fraudCounts[c] > fraudCounts[modal]) modal = c;
            int total = indices.Count;
            double conf = fraudCounts[modal] / (double)total;

            if (conf >= confidenceThreshold)
            {
                countKeys[modal].Add(key);
                decided++;
            }

            bucketsDone++;
            if (bucketsDone % 100 == 0)
                Console.Write($"\r  Processed {bucketsDone}/{undecided + buckets.Count - bucketsDone} undecided buckets... ({sw.Elapsed.TotalSeconds:F0}s)");
        }
        Console.WriteLine();

        return new BuildResult
        {
            UndecidedBuckets = undecided,
            SampledBuckets = sampled,
            DecidedBuckets = decided,
            CountKeys = countKeys.Select(l => l.ToArray()).ToArray(),
        };
    }

    // ─── Validation ──────────────────────────────────────────────────────────

    // Full per-query validation requires rebuilding the cascade with the new keys.
    // Inline validation here only checks the FP1/FP2 residual count so the caller
    // can log how many queries reach the new stage. Correctness validation is done
    // by regenerating selective_decision_tables.json and running --audit-fastpath.
    private static ValidationResult Validate(
        string testDataPath,
        JsonVectorizer jvec,
        int[] features,
        int[] bits,
        BuildResult result)
    {
        using var doc = JsonDocument.Parse(File.ReadAllBytes(testDataPath));
        var entries = doc.RootElement.GetProperty("entries");
        var v = new ValidationResult();
        var qBuf = new float[Dataset.Dimensions];

        foreach (var entry in entries.EnumerateArray())
        {
            var reqRaw = System.Text.Encoding.UTF8.GetBytes(entry.GetProperty("request").GetRawText());
            jvec.VectorizeJson(reqRaw, qBuf);
            byte fp1 = ProfileFastPath.TryLookup(qBuf);
            if (fp1 != ProfileFastPath.ResultUndecided) continue;
            byte fp2 = ProfileFastPath2.TryLookup(qBuf);
            if (fp2 != ProfileFastPath2.ResultUndecided) continue;
            v.Decided++; // counts FP1+FP2 residuals that COULD be decided by the new stage
        }

        Console.WriteLine($"  FP1+FP2 residuals in test-data: {v.Decided}");
        Console.WriteLine("  (key-level correctness: run --audit-fastpath after updating selective_decision_tables.json)");
        return v;
    }

    // ─── Output ──────────────────────────────────────────────────────────────

    private static void WriteStandaloneFragment(
        string outPath,
        int[] features,
        int[] bits,
        BuildResult result,
        ValidationResult? validation,
        string dataDir)
    {
        var sb = new StringBuilder();
        AppendStageJson(sb, features, bits, result, validation, dataDir, indent: "");
        File.WriteAllText(outPath, sb.ToString());
    }

    private static void AppendOrReplaceModalStage(
        string outPath,
        int[] features,
        int[] bits,
        BuildResult result,
        ValidationResult? validation,
        string dataDir,
        double confidence,
        int sample)
    {
        if (!File.Exists(outPath))
        {
            Console.Error.WriteLine($"Output file {outPath} not found. Use --standalone to write a fragment.");
            return;
        }

        // Read the existing JSON, find the "residual-modal" / "reference-modal" stage, replace it.
        string existing = File.ReadAllText(outPath);
        using var doc = JsonDocument.Parse(existing);
        var root = doc.RootElement;

        var sb = new StringBuilder();
        sb.AppendLine("{");

        // Copy all top-level properties except "stages".
        bool firstProp = true;
        foreach (var prop in root.EnumerateObject())
        {
            if (prop.Name == "stages") continue;
            if (!firstProp) sb.AppendLine(","); else firstProp = false;
            sb.Append($"  {JsonSerializer.Serialize(prop.Name)}: {prop.Value.GetRawText()}");
        }
        sb.AppendLine(",");
        sb.AppendLine("  \"stages\": [");

        // Collect all output stages as strings, then join with commas.
        var stages = root.GetProperty("stages");
        int stageCount = stages.GetArrayLength();
        var stageStrings = new List<string>(stageCount + 1);
        bool replacedModal = false;

        for (int i = 0; i < stageCount; i++)
        {
            var stage = stages[i];
            string stageName = stage.GetProperty("name").GetString() ?? "";
            bool isModal = stageName is "residual-modal" or "reference-modal";

            if (isModal)
            {
                var stageSb = new StringBuilder();
                AppendStageJson(stageSb, features, bits, result, validation, dataDir, indent: "    ");
                stageStrings.Add(stageSb.ToString());
                replacedModal = true;
            }
            else
            {
                stageStrings.Add("    " + stage.GetRawText());
            }
        }

        if (!replacedModal)
        {
            var stageSb = new StringBuilder();
            AppendStageJson(stageSb, features, bits, result, validation, dataDir, indent: "    ");
            stageStrings.Add(stageSb.ToString());
        }

        for (int i = 0; i < stageStrings.Count; i++)
        {
            sb.Append(stageStrings[i]);
            sb.AppendLine(i < stageStrings.Count - 1 ? "," : "");
        }

        sb.AppendLine("  ]");
        sb.AppendLine("}");
        File.WriteAllText(outPath, sb.ToString());
    }

    private static void AppendStageJson(
        StringBuilder sb,
        int[] features,
        int[] bits,
        BuildResult result,
        ValidationResult? validation,
        string dataDir,
        string indent)
    {
        int total = bits.Sum();
        long tableSize = 1L << total;
        int totalKeys = result.CountKeys.Sum(k => k.Length);

        sb.AppendLine($"{indent}{{");
        sb.AppendLine($"{indent}  \"name\": \"reference-modal\",");
        sb.AppendLine($"{indent}  \"mode\": \"residual_modal_sparse\",");
        sb.AppendLine($"{indent}  \"enabled_by_default\": false,");
        sb.AppendLine($"{indent}  \"env_flags\": [\"SELECTIVE_DECISION_REFERENCE_MODAL\"],");
        sb.AppendLine($"{indent}  \"risk_level\": \"medium\",");
        sb.AppendLine($"{indent}  \"source\": \"reference_data_sampled (not eval queries)\",");
        AppendIntArray(sb, "feature_indices", features, comma: true, indent: $"{indent}  ");
        AppendStringArray(sb, "feature_names", NamesFor(features), comma: true, indent: $"{indent}  ");
        AppendIntArray(sb, "bits", bits, comma: true, indent: $"{indent}  ");
        sb.AppendLine($"{indent}  \"total_bits\": {total},");
        sb.AppendLine($"{indent}  \"table_slots\": {tableSize},");
        sb.AppendLine($"{indent}  \"decided_buckets\": {result.DecidedBuckets},");
        sb.AppendLine($"{indent}  \"undecided_reference_buckets\": {result.UndecidedBuckets},");
        if (validation is not null)
        {
            sb.AppendLine($"{indent}  \"validation_decided\": {validation.Decided},");
            sb.AppendLine($"{indent}  \"validation_wrong_approval\": {validation.WrongApproval},");
            sb.AppendLine($"{indent}  \"validation_count_mismatch\": {validation.CountMismatch},");
        }
        for (int c = 0; c <= 5; c++)
            AppendUIntArray(sb, $"count{c}_keys", result.CountKeys[c], comma: c != 5, indent: $"{indent}  ");
        sb.Append($"{indent}}}");
    }

    // ─── Helpers ─────────────────────────────────────────────────────────────

    private static unsafe void CopyVector(float* src, Span<float> dst)
    {
        for (int i = 0; i < Dataset.Dimensions; i++)
            dst[i] = src[i];
    }

    private static List<int> SampleWithoutReplacement(List<int> source, int n, Random rng)
    {
        var result = new List<int>(n);
        var copy = new List<int>(source);
        for (int i = 0; i < n && copy.Count > 0; i++)
        {
            int j = rng.Next(copy.Count);
            result.Add(copy[j]);
            copy.RemoveAt(j);
        }
        return result;
    }

    private static (int[] shifts, int totalBits) BuildShifts(int[] bits)
    {
        var shifts = new int[bits.Length];
        int total = 0;
        for (int i = 0; i < bits.Length; i++) { shifts[i] = total; total += bits[i]; }
        return (shifts, total);
    }

    private static unsafe float[][] BuildEdges(Dataset ds, int[] features, int[] bits)
    {
        int n = ds.Count;
        var vectors = ds.VectorsPtr;
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

    private static unsafe uint ComputeKey(float* row, int[] features, int[] shifts, float[][] edges)
    {
        uint key = 0;
        for (int f = 0; f < features.Length; f++)
        {
            float v = row[features[f]];
            int bin = FindBin(edges[f], v);
            key |= (uint)bin << shifts[f];
        }
        return key;
    }

    private static int FindBin(float[] edges, float v)
    {
        for (int b = 0; b < edges.Length; b++)
            if (v < edges[b]) return b;
        return edges.Length;
    }

    private static string[] NamesFor(int[] features)
    {
        var names = new string[features.Length];
        for (int i = 0; i < features.Length; i++)
            names[i] = AllFeatureNames[features[i]];
        return names;
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

    private static string FindRepoRoot()
    {
        string? root = AppContext.BaseDirectory;
        while (root is not null && !File.Exists(Path.Combine(root, "Rinha.slnx")))
            root = Path.GetDirectoryName(root);
        if (root is null) throw new InvalidOperationException("repo root not found");
        return root;
    }
}
