using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Diagnosers;
using BenchmarkDotNet.Jobs;
using BenchmarkDotNet.Running;
using Rinha.Api;

if (args.Length > 0 && args[0] == "--recall")
    return RecallCheck.Run(args[1..]);

var config = ManualConfig.CreateMinimumViable()
    .AddJob(Job.Default)
    .AddDiagnoser(MemoryDiagnoser.Default);

BenchmarkRunner.Run<ScorerBenchmarks>(config, args);
return 0;

static class RecallCheck
{
    /// <summary>
    /// Compares the IVF scorer's approve/reject decision and raw score against the
    /// brute-force ground truth across N synthetic queries (deterministic seed).
    /// Exits 0 if the disagreement rate is &lt;= tolerance (default 1%), 1 otherwise.
    /// Usage: Rinha.Bench --recall [--n=2000] [--tol=0.01] [--seed=42] [--rerank=32] [--nprobe=96] [--early-stop-pct=75]
    /// </summary>
    public static int Run(string[] args)
    {
        int n = 2000;
        double tol = 0.01;
        int seed = 42;
        int rerank = 32;
        int nprobe = 96;
        int earlyStopPct = 75;
        bool earlyStop = true;
        foreach (var a in args)
        {
            if (a.StartsWith("--n=")) n = int.Parse(a[4..]);
            else if (a.StartsWith("--tol=")) tol = double.Parse(a[6..], System.Globalization.CultureInfo.InvariantCulture);
            else if (a.StartsWith("--seed=")) seed = int.Parse(a[7..]);
            else if (a.StartsWith("--rerank=")) rerank = int.Parse(a[9..]);
            else if (a.StartsWith("--nprobe=")) nprobe = int.Parse(a[9..]);
            else if (a.StartsWith("--early-stop-pct=")) earlyStopPct = int.Parse(a[17..]);
            else if (a == "--no-early-stop") earlyStop = false;
        }

        var root = FindRepoRoot();
        var vec = Path.Combine(root, "data", "references.bin");
        var lab = Path.Combine(root, "data", "labels.bin");
        var q8  = Path.Combine(root, "data", "references_q8.bin");
        var cents = Path.Combine(root, "data", "ivf_centroids.bin");
        var offs  = Path.Combine(root, "data", "ivf_offsets.bin");

        if (!File.Exists(vec) || !File.Exists(cents))
        {
            Console.Error.WriteLine("Recall check requires data/references.bin, data/ivf_centroids.bin and friends.");
            Console.Error.WriteLine("Run preprocessing first (docker compose --profile dataprep up).");
            return 2;
        }

        using var dataset = Dataset.Open(vec, lab,
            File.Exists(q8) ? q8 : null,
            null,
            cents,
            File.Exists(offs) ? offs : null);

        var brute = new Rinha.Api.Scorers.BruteForceScorer(dataset);
        var ivf = new Rinha.Api.Scorers.IvfScorer(dataset, nProbe: nprobe, kPrime: rerank,
            earlyStop: earlyStop, earlyStopPct: earlyStopPct);

        var rng = new Random(seed);
        Span<float> q = stackalloc float[Dataset.Dimensions];
        int approveDisagree = 0;
        double scoreDiffSum = 0;
        double scoreDiffMax = 0;
        const float threshold = 0.6f;
        var sw = System.Diagnostics.Stopwatch.StartNew();
        for (int i = 0; i < n; i++)
        {
            for (int d = 0; d < Dataset.Dimensions; d++)
                q[d] = (float)(rng.NextDouble() * 2 - 1);
            var sb = brute.Score(q);
            var si = ivf.Score(q);
            bool ab = sb < threshold;
            bool ai = si < threshold;
            if (ab != ai) approveDisagree++;
            double diff = Math.Abs(sb - si);
            scoreDiffSum += diff;
            if (diff > scoreDiffMax) scoreDiffMax = diff;
        }
        sw.Stop();

        double rate = approveDisagree / (double)n;
        Console.WriteLine($"recall-check: N={n} approve-disagreements={approveDisagree} ({rate:P3}) " +
                          $"score-diff avg={scoreDiffSum / n:F4} max={scoreDiffMax:F4} " +
                          $"elapsed={sw.ElapsedMilliseconds}ms");
        if (rate > tol)
        {
            Console.Error.WriteLine($"FAIL: disagreement rate {rate:P3} > tol {tol:P3}");
            return 1;
        }
        Console.WriteLine("PASS");
        return 0;
    }

    private static string FindRepoRoot()
    {
        var dir = AppContext.BaseDirectory;
        while (dir is not null && !File.Exists(Path.Combine(dir, "Rinha.slnx")))
            dir = Path.GetDirectoryName(dir);
        return dir ?? throw new InvalidOperationException("Could not locate repo root");
    }
}

[MemoryDiagnoser]
public class ScorerBenchmarks
{
    private Dataset _dataset = default!;
    private IFraudScorer _scorer = default!;
    private float[] _query = default!;

    [Params("brute", "q8")]
    public string Scorer { get; set; } = "brute";

    [GlobalSetup]
    public void Setup()
    {
        var root = FindRepoRoot();
        var vec = Path.Combine(root, "data", "references.bin");
        var lab = Path.Combine(root, "data", "labels.bin");
        var q8  = Path.Combine(root, "data", "references_q8.bin");
        _dataset = Dataset.Open(vec, lab, File.Exists(q8) ? q8 : null);
        _scorer = ScorerFactory.Create(Scorer, _dataset);

        // Representative query (first example payload, hand-vectorized).
        _query = new float[Dataset.Dimensions];
        _query[0] = 0.0041f; _query[1] = 0.1667f; _query[2] = 0.05f;  _query[3] = 0.7826f;
        _query[4] = 0.3333f; _query[5] = -1f;     _query[6] = -1f;    _query[7] = 0.0292f;
        _query[8] = 0.15f;   _query[9] = 0f;      _query[10] = 1f;    _query[11] = 0f;
        _query[12] = 0.15f;  _query[13] = 0.006f;
    }

    [GlobalCleanup]
    public void Cleanup() => _dataset.Dispose();

    [Benchmark]
    public float Score() => _scorer.Score(_query);

    private static string FindRepoRoot()
    {
        var dir = AppContext.BaseDirectory;
        while (dir is not null && !File.Exists(Path.Combine(dir, "Rinha.slnx")))
            dir = Path.GetDirectoryName(dir);
        return dir ?? throw new InvalidOperationException("Could not locate repo root");
    }
}
