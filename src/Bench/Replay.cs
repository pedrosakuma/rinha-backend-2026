using System.Diagnostics;
using System.Globalization;
using System.Text;
using System.Text.Json;
using Rinha.Api;
using Rinha.Api.Scorers;

namespace Rinha.Bench;

/// <summary>
/// Offline replay harness: vectorizes every entry from a real test-data.json,
/// scores it with brute-force (ground truth) and the IVF scorer under various
/// parameter combinations, and reports correctness deltas (FN/FP, score
/// disagreement) without any HTTP/network overhead.
///
/// Usage:
///   Rinha.Bench --replay --test-data=/tmp/rinha-eval/test/test-data.json [--limit=N] \
///       [--data-dir=./data] [--grid] [--config=KP=128,NP=8,BNP=32,BKP=128]
///
/// Without --grid, runs the active default config from the env vars (mirrors
/// production). With --grid, sweeps a small predefined parameter grid focused
/// on borderline behavior.
/// </summary>
public static class Replay
{
    private record Cfg(int NProbe, int KPrime, int BorderlineNProbe, int BorderlineKPrime, bool BboxGuided = true, bool BboxRepair = false, bool EarlyStop = true, int EarlyStopPct = 75, bool Q16 = true)
    {
        public override string ToString()
            => $"NP={NProbe,3} KP={KPrime,4} BNP={BorderlineNProbe,3} BKP={BorderlineKPrime,4} ES={(EarlyStop ? 1 : 0)} ESPCT={EarlyStopPct,2} Q16={(Q16 ? 1 : 0)} BBG={(BboxGuided ? 1 : 0)} BBR={(BboxRepair ? 1 : 0)}";
    }

    public static int Run(string[] args)
    {
        string testData = "/tmp/rinha-eval/test/test-data.json";
        string dataDir = Path.Combine(FindRepoRoot(), "data");
        int limit = int.MaxValue;
        bool grid = false;
        string? configArg = null;
        bool csv = false;
        bool candidatesOnly = false;
        float candLow = 0.30f, candHigh = 0.80f;
        float jitter = 0f;
        int jitterSeed = 42;
        int randomN = 0;
        int randomSeed = 7;
        int edgeN = 0;
        int edgeSeed = 13;

        foreach (var a in args)
        {
            if (a.StartsWith("--test-data=")) testData = a[12..];
            else if (a.StartsWith("--data-dir=")) dataDir = a[11..];
            else if (a.StartsWith("--limit=")) limit = int.Parse(a[8..]);
            else if (a == "--grid") grid = true;
            else if (a.StartsWith("--config=")) configArg = a[9..];
            else if (a == "--csv") csv = true;
            else if (a == "--candidates-only") candidatesOnly = true;
            else if (a.StartsWith("--cand-low=")) candLow = float.Parse(a[11..], CultureInfo.InvariantCulture);
            else if (a.StartsWith("--cand-high=")) candHigh = float.Parse(a[12..], CultureInfo.InvariantCulture);
            else if (a.StartsWith("--jitter=")) jitter = float.Parse(a[9..], CultureInfo.InvariantCulture);
            else if (a.StartsWith("--jitter-seed=")) jitterSeed = int.Parse(a[14..]);
            else if (a.StartsWith("--random=")) randomN = int.Parse(a[9..]);
            else if (a.StartsWith("--random-seed=")) randomSeed = int.Parse(a[14..]);
            else if (a.StartsWith("--edge=")) edgeN = int.Parse(a[7..]);
            else if (a.StartsWith("--edge-seed=")) edgeSeed = int.Parse(a[12..]);
        }

        var vec = Path.Combine(dataDir, "references.bin");
        var lab = Path.Combine(dataDir, "labels.bin");
        var q8 = Path.Combine(dataDir, "references_q8.bin");
        var q8Soa = Path.Combine(dataDir, "references_q8_soa.bin");
        var q16 = Path.Combine(dataDir, "references_q16.bin");
        var cents = Path.Combine(dataDir, "ivf_centroids.bin");
        var offs = Path.Combine(dataDir, "ivf_offsets.bin");
        var bbmin = Path.Combine(dataDir, "ivf_bbox_min.bin");
        var bbmax = Path.Combine(dataDir, "ivf_bbox_max.bin");

        if (!File.Exists(vec))
        {
            Console.Error.WriteLine($"Missing data files in {dataDir}. Extract from API image first.");
            return 2;
        }

        using var dataset = Dataset.Open(vec, lab,
            File.Exists(q8) ? q8 : null,
            File.Exists(q8Soa) ? q8Soa : null,
            File.Exists(q16) ? q16 : null,
            cents,
            File.Exists(offs) ? offs : null,
            File.Exists(bbmin) ? bbmin : null,
            File.Exists(bbmax) ? bbmax : null);

        var root = FindRepoRoot();
        var norm = NormalizationConstants.Load(Path.Combine(root, "resources/normalization.json"));
        var mcc = MccRiskTable.Load(Path.Combine(root, "resources/mcc_risk.json"));
        var jvec = new JsonVectorizer(norm, mcc);

        Console.Error.WriteLine($"Loading {testData} ...");
        var queries = LoadQueries(testData, jvec, limit);
        Console.Error.WriteLine($"Loaded {queries.Count} queries.");

        // Stress: optionally synthesize / perturb queries for robustness testing (#3).
        // Cache key incorporates stress params so each variant has its own GT cache.
        string stressTag = "";
        if (jitter > 0f)
        {
            queries = ApplyJitter(queries, jitter, jitterSeed);
            stressTag = $"-jitter{jitter:F3}-s{jitterSeed}";
            Console.Error.WriteLine($"Jittered {queries.Count} queries with σ={jitter} seed={jitterSeed}");
        }
        if (randomN > 0)
        {
            queries = SynthRandom(randomN, randomSeed);
            stressTag = $"-random{randomN}-s{randomSeed}";
            Console.Error.WriteLine($"Synthetic random {queries.Count} queries seed={randomSeed}");
        }
        if (edgeN > 0)
        {
            queries = SynthEdge(dataset, edgeN, edgeSeed);
            stressTag = $"-edge{edgeN}-s{edgeSeed}";
            Console.Error.WriteLine($"Edge-case {queries.Count} queries seed={edgeSeed}");
        }

        var brute = new BruteForceScorer(dataset);
        const float thr = 0.6f;

        // Ground truth (brute-force) — cache to disk so we don't pay 10min/run.
        var gtPath = Path.Combine(dataDir, $".gt-cache-{queries.Count}{stressTag}.bin");
        var gtScore = new float[queries.Count];
        var gtApproved = new bool[queries.Count];
        if (File.Exists(gtPath) && new FileInfo(gtPath).Length == queries.Count * sizeof(float))
        {
            Console.Error.WriteLine($"Loading cached GT from {gtPath} ...");
            var raw = File.ReadAllBytes(gtPath);
            Buffer.BlockCopy(raw, 0, gtScore, 0, raw.Length);
            for (int i = 0; i < queries.Count; i++) gtApproved[i] = gtScore[i] < thr;
        }
        else
        {
            Console.Error.WriteLine($"Computing brute-force ground truth (parallel, {Environment.ProcessorCount} threads) ...");
            var sw = Stopwatch.StartNew();
            // Each thread builds its own scorer (state-free, but instances are cheap and
            // avoid any sharing concerns).
            var queriesArr = queries;
            Parallel.For(0, queriesArr.Count, () => new BruteForceScorer(dataset),
                (i, _, local) =>
                {
                    var s = local.Score(queriesArr[i].Vec);
                    gtScore[i] = s;
                    gtApproved[i] = s < thr;
                    return local;
                },
                _ => { });
            sw.Stop();
            Console.Error.WriteLine($"Brute done in {sw.ElapsedMilliseconds}ms.");
            var raw = new byte[queries.Count * sizeof(float)];
            Buffer.BlockCopy(gtScore, 0, raw, 0, raw.Length);
            File.WriteAllBytes(gtPath, raw);
            Console.Error.WriteLine($"Cached GT to {gtPath}.");
        }
        var swT = Stopwatch.StartNew();

        // Optional: restrict to borderline candidate set (queries with brute score
        // in [candLow, candHigh]). Speeds up grid sweep enormously while still
        // testing every query that could plausibly flip approve/reject.
        int[] candIdx;
        if (candidatesOnly)
        {
            var list = new List<int>();
            for (int i = 0; i < queries.Count; i++)
                if (gtScore[i] >= candLow && gtScore[i] <= candHigh) list.Add(i);
            candIdx = list.ToArray();
            Console.Error.WriteLine($"Candidate set: {candIdx.Length}/{queries.Count} queries with GT score in [{candLow:F2},{candHigh:F2}]");
        }
        else
        {
            candIdx = new int[queries.Count];
            for (int i = 0; i < queries.Count; i++) candIdx[i] = i;
        }

        // Configs to evaluate.
        var configs = new List<Cfg>();
        if (configArg is not null)
        {
            configs.Add(ParseConfig(configArg));
        }
        else if (grid)
        {
            // Focus: borderline params (BKP, BNP) for L8 tuning.
            int[] bkps = { 0, 64, 128, 192, 256, 384 };
            int[] bnps = { 16, 32, 48, 64 };
            int[] kps = { 32, 48 };
            foreach (var kp in kps)
                foreach (var bkp in bkps)
                    foreach (var bnp in bnps)
                        configs.Add(new Cfg(NProbe: 8, KPrime: kp, BorderlineNProbe: bnp, BorderlineKPrime: bkp));
        }
        else
        {
            // Single run — current production config.
            configs.Add(new Cfg(NProbe: 8, KPrime: 32, BorderlineNProbe: 32, BorderlineKPrime: 128));
        }

        if (csv)
            Console.WriteLine("nprobe,kprime,bnp,bkp,fn,fp,disagree,maxScoreDiff,avgScoreDiff,elapsed_ms");
        else
            Console.WriteLine($"{"config",-58} {"FN",4} {"FP",4} {"disagree",8} {"avgΔ",8} {"maxΔ",8} {"ms",8}");

        foreach (var cfg in configs)
        {
            // Inject env so IvfScorer picks them up.
            Environment.SetEnvironmentVariable("IVF_BORDERLINE_NPROBE", cfg.BorderlineNProbe.ToString());
            Environment.SetEnvironmentVariable("IVF_BORDERLINE_RERANK", cfg.BorderlineKPrime.ToString());
            Environment.SetEnvironmentVariable("IVF_Q16", cfg.Q16 ? "1" : "0");
            var ivf = new IvfScorer(dataset,
                nProbe: cfg.NProbe,
                kPrime: cfg.KPrime,
                earlyStop: cfg.EarlyStop,
                earlyStopPct: cfg.EarlyStopPct,
                bboxRepair: cfg.BboxRepair,
                bboxGuided: cfg.BboxGuided);

            int fn = 0, fp = 0, disagree = 0;
            int firstFnIdx = -1;
            double diffSum = 0, diffMax = 0;
            var perQuery = new long[candIdx.Length];
            var swQ = new System.Diagnostics.Stopwatch();
            swT.Restart();
            for (int qi = 0; qi < candIdx.Length; qi++)
            {
                int i = candIdx[qi];
                swQ.Restart();
                var s = ivf.Score(queries[i].Vec);
                swQ.Stop();
                perQuery[qi] = swQ.ElapsedTicks;
                bool approved = s < thr;
                if (approved != gtApproved[i])
                {
                    disagree++;
                    if (gtApproved[i] && !approved) fp++;
                    else { if (firstFnIdx < 0) firstFnIdx = i; fn++; }
                }
                double diff = Math.Abs(s - gtScore[i]);
                diffSum += diff;
                if (diff > diffMax) diffMax = diff;
            }
            swT.Stop();

            Array.Sort(perQuery);
            double tickToUs = 1_000_000.0 / System.Diagnostics.Stopwatch.Frequency;
            double p50 = perQuery[(int)(perQuery.Length * 0.50)] * tickToUs;
            double p99 = perQuery[(int)(perQuery.Length * 0.99)] * tickToUs;
            double p999 = perQuery[(int)(perQuery.Length * 0.999)] * tickToUs;

            double avg = diffSum / candIdx.Length;
            if (csv)
                Console.WriteLine($"{cfg.NProbe},{cfg.KPrime},{cfg.BorderlineNProbe},{cfg.BorderlineKPrime},{fn},{fp},{disagree},{diffMax:F4},{avg:F4},{swT.ElapsedMilliseconds},{p50:F1},{p99:F1},{p999:F1}");
            else
                Console.WriteLine($"{cfg,-86} {fn,4} {fp,4} {disagree,8} {avg,7:F4} {diffMax,7:F4} {swT.ElapsedMilliseconds,7}ms p50={p50,5:F1}us p99={p99,6:F1}us p999={p999,6:F1}us{(firstFnIdx >= 0 ? $"  firstFn=#{firstFnIdx} id={queries[firstFnIdx].Id}" : "")}");
        }

        return 0;
    }

    private record QueryRow(string Id, float[] Vec, bool ExpectedApproved, float ExpectedScore);

    private static List<QueryRow> LoadQueries(string path, JsonVectorizer jvec, int limit)
    {
        var bytes = File.ReadAllBytes(path);
        using var doc = JsonDocument.Parse(bytes);
        var entries = doc.RootElement.GetProperty("entries");
        var list = new List<QueryRow>(Math.Min(entries.GetArrayLength(), limit));
        var vecBuf = new float[Dataset.Dimensions];
        int n = 0;
        foreach (var entry in entries.EnumerateArray())
        {
            if (n >= limit) break;
            var req = entry.GetProperty("request");
            var raw = Encoding.UTF8.GetBytes(req.GetRawText());
            var vec = new float[Dataset.Dimensions];
            jvec.VectorizeJson(raw, vec);
            string id = req.TryGetProperty("id", out var idEl) ? idEl.GetString() ?? "" : "";
            bool approved = entry.TryGetProperty("expected_approved", out var ap) && ap.GetBoolean();
            float score = entry.TryGetProperty("expected_fraud_score", out var sc) ? sc.GetSingle() : 0f;
            list.Add(new QueryRow(id, vec, approved, score));
            n++;
        }
        return list;
    }

    private static Cfg ParseConfig(string s)
    {
        int np = 8, kp = 32, bnp = 32, bkp = 0;
        bool bbg = true;
        bool bbr = false;
        int espct = 75;
        bool q16 = true;
        foreach (var part in s.Split(','))
        {
            var kv = part.Split('=');
            if (kv.Length != 2) continue;
            int v = int.Parse(kv[1], CultureInfo.InvariantCulture);
            switch (kv[0].Trim().ToUpperInvariant())
            {
                case "NP": case "NPROBE": np = v; break;
                case "KP": case "KPRIME": kp = v; break;
                case "BNP": case "BORDERLINE_NPROBE": bnp = v; break;
                case "BKP": case "BORDERLINE_RERANK": case "BORDERLINE_KPRIME": bkp = v; break;
                case "BBG": case "BBOX_GUIDED": bbg = v != 0; break;
                case "BBR": case "BBOX_REPAIR": bbr = v != 0; break;
                case "ESPCT": case "EARLY_STOP_PCT": espct = v; break;
                case "Q16": q16 = v != 0; break;
            }
        }
        return new Cfg(np, kp, bnp, bkp, BboxGuided: bbg, BboxRepair: bbr, EarlyStopPct: espct, Q16: q16);
    }

    private static string FindRepoRoot()
    {
        var dir = AppContext.BaseDirectory;
        while (dir is not null && !File.Exists(Path.Combine(dir, "Rinha.slnx")))
            dir = Path.GetDirectoryName(dir);
        return dir ?? throw new InvalidOperationException("Could not locate repo root");
    }

    // -------------------- Stress distributions (#3) --------------------
    // Box-Muller Normal(0,1) sample.
    private static float NextGaussian(Random rng)
    {
        double u1 = 1.0 - rng.NextDouble();
        double u2 = 1.0 - rng.NextDouble();
        return (float)(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2));
    }

    /// <summary>
    /// Per-query Normal(0, σ) perturbation on every dimension. Deterministic per
    /// query index by using a per-query Random seeded with (globalSeed ^ index).
    /// Output features are clamped to [0,1] (most preprocessor outputs live there).
    /// </summary>
    private static List<QueryRow> ApplyJitter(List<QueryRow> input, float sigma, int seed)
    {
        var output = new List<QueryRow>(input.Count);
        for (int i = 0; i < input.Count; i++)
        {
            var rng = new Random(seed ^ unchecked((int)(i * 2654435761L)));
            var v = new float[Dataset.Dimensions];
            var src = input[i].Vec;
            for (int d = 0; d < Dataset.Dimensions; d++)
            {
                float val = src[d] + sigma * NextGaussian(rng);
                if (val < 0f) val = 0f;
                else if (val > 1f) val = 1f;
                v[d] = val;
            }
            output.Add(new QueryRow(input[i].Id + ".jit", v, false, 0f));
        }
        return output;
    }

    /// <summary>
    /// Synthetic uniform queries in [0,1]^D. Stress-tests degenerate configs that
    /// only work on realistic distributions.
    /// </summary>
    private static List<QueryRow> SynthRandom(int n, int seed)
    {
        var rng = new Random(seed);
        var output = new List<QueryRow>(n);
        for (int i = 0; i < n; i++)
        {
            var v = new float[Dataset.Dimensions];
            for (int d = 0; d < Dataset.Dimensions; d++) v[d] = (float)rng.NextDouble();
            output.Add(new QueryRow($"rnd-{i}", v, false, 0f));
        }
        return output;
    }

    /// <summary>
    /// Edge-case queries: linear interpolation between two random centroid pairs at
    /// t∈[0.4, 0.6]. These maximize the chance of borderline IVF gates firing
    /// (queries equidistant to two cluster centers).
    /// </summary>
    private static unsafe List<QueryRow> SynthEdge(Dataset dataset, int n, int seed)
    {
        var rng = new Random(seed);
        var output = new List<QueryRow>(n);
        int nlist = dataset.NumCells;
        var centBase = dataset.CentroidsPtr;
        for (int i = 0; i < n; i++)
        {
            int a = rng.Next(nlist);
            int b = rng.Next(nlist);
            while (b == a) b = rng.Next(nlist);
            float t = 0.4f + 0.2f * (float)rng.NextDouble();
            var v = new float[Dataset.Dimensions];
            float* ca = centBase + (long)a * Dataset.PaddedDimensions;
            float* cb = centBase + (long)b * Dataset.PaddedDimensions;
            for (int d = 0; d < Dataset.Dimensions; d++)
            {
                float val = (1f - t) * ca[d] + t * cb[d];
                if (val < 0f) val = 0f;
                else if (val > 1f) val = 1f;
                v[d] = val;
            }
            output.Add(new QueryRow($"edge-{a}-{b}-t{t:F2}", v, false, 0f));
        }
        return output;
    }
}
