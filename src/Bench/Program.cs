using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Diagnosers;
using BenchmarkDotNet.Jobs;
using BenchmarkDotNet.Running;
using Rinha.Api;

var config = ManualConfig.CreateMinimumViable()
    .AddJob(Job.Default)
    .AddDiagnoser(MemoryDiagnoser.Default);

BenchmarkRunner.Run<ScorerBenchmarks>(config, args);

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
