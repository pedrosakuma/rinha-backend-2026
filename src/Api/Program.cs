using Rinha.Api;

var builder = WebApplication.CreateSlimBuilder(args);

builder.Logging.ClearProviders();
builder.WebHost.ConfigureKestrel(options =>
{
    options.AddServerHeader = false;
    options.AllowSynchronousIO = false;
    var udsPath = Environment.GetEnvironmentVariable("UDS_PATH");
    if (!string.IsNullOrEmpty(udsPath))
    {
        // J11a: listen on a Unix Domain Socket so the LB upstream is local FS, not TCP loopback.
        // Saves ~30-50us per request (no TCP/IP stack, no port allocation, no Nagle delays).
        if (File.Exists(udsPath)) File.Delete(udsPath);
        options.ListenUnixSocket(udsPath);
    }
    else
    {
        var port = int.Parse(Environment.GetEnvironmentVariable("PORT") ?? "9999");
        options.ListenAnyIP(port);
    }
});

builder.Services.ConfigureHttpJsonOptions(o =>
{
    o.SerializerOptions.TypeInfoResolverChain.Insert(0, AppJsonContext.Default);
});

var vectorsPath = Environment.GetEnvironmentVariable("VECTORS_PATH") ?? "/data/references.bin";
var labelsPath = Environment.GetEnvironmentVariable("LABELS_PATH") ?? "/data/labels.bin";
var vectorsQ8Path = Environment.GetEnvironmentVariable("VECTORS_Q8_PATH"); // optional
var ivfCentroidsPath = Environment.GetEnvironmentVariable("IVF_CENTROIDS_PATH");
var ivfOffsetsPath = Environment.GetEnvironmentVariable("IVF_OFFSETS_PATH");
var ivfBboxMinPath = Environment.GetEnvironmentVariable("IVF_BBOX_MIN_PATH");
var ivfBboxMaxPath = Environment.GetEnvironmentVariable("IVF_BBOX_MAX_PATH");
var pqCodebooksPath = Environment.GetEnvironmentVariable("PQ_CODEBOOKS_PATH");
var pqCodesPath = Environment.GetEnvironmentVariable("PQ_CODES_PATH");
var pqM = int.TryParse(Environment.GetEnvironmentVariable("PQ_M"), out var _pqM) ? _pqM : 7;
var pqKsub = int.TryParse(Environment.GetEnvironmentVariable("PQ_KSUB"), out var _pqK) ? _pqK : 256;
var mccRiskPath = Environment.GetEnvironmentVariable("MCC_RISK_PATH") ?? "/app/resources/mcc_risk.json";
var normalizationPath = Environment.GetEnvironmentVariable("NORMALIZATION_PATH") ?? "/app/resources/normalization.json";

var normalization = NormalizationConstants.Load(normalizationPath);
var mccRisk = MccRiskTable.Load(mccRiskPath);
var dataset = Dataset.Open(vectorsPath, labelsPath, vectorsQ8Path, ivfCentroidsPath, ivfOffsetsPath, ivfBboxMinPath, ivfBboxMaxPath, pqCodebooksPath, pqCodesPath, pqM, pqKsub);
var vectorizer = new Vectorizer(normalization, mccRisk);
var scorerName = Environment.GetEnvironmentVariable("SCORER") ?? "brute";
IFraudScorer scorer = ScorerFactory.Create(scorerName, dataset);

var app = builder.Build();

app.MapGet("/ready", () => Results.Ok());

var profile = Environment.GetEnvironmentVariable("PROFILE_TIMING") == "1";
if (profile)
{
    long n = 0, vSum = 0, sSum = 0, jSum = 0;
    long vMax = 0, sMax = 0, jMax = 0;
    var lockObj = new object();
    const int Window = 5000;
    app.MapPost("/fraud-score", (FraudRequest request) =>
    {
        long t0 = System.Diagnostics.Stopwatch.GetTimestamp();
        Span<float> query = stackalloc float[Dataset.Dimensions];
        vectorizer.Vectorize(request, query);
        long t1 = System.Diagnostics.Stopwatch.GetTimestamp();
        var score = scorer.Score(query);
        long t2 = System.Diagnostics.Stopwatch.GetTimestamp();
        var resp = Results.Json(new FraudResponse(score < 0.6f, score), AppJsonContext.Default.FraudResponse);
        long t3 = System.Diagnostics.Stopwatch.GetTimestamp();

        long v = t1 - t0, s = t2 - t1, j = t3 - t2;
        bool flush = false;
        lock (lockObj)
        {
            n++;
            vSum += v; sSum += s; jSum += j;
            if (v > vMax) vMax = v;
            if (s > sMax) sMax = s;
            if (j > jMax) jMax = j;
            if (n >= Window) flush = true;
        }
        if (flush)
        {
            long N, vS, sS, jS, vM, sM, jM;
            lock (lockObj)
            {
                N = n; vS = vSum; sS = sSum; jS = jSum; vM = vMax; sM = sMax; jM = jMax;
                n = 0; vSum = sSum = jSum = 0; vMax = sMax = jMax = 0;
            }
            double f = 1e6 / System.Diagnostics.Stopwatch.Frequency;
            Console.WriteLine(
                $"[timing N={N}] vec={vS*f/N:F1}us(max {vM*f:F1}) " +
                $"score={sS*f/N:F1}us(max {sM*f:F1}) " +
                $"json={jS*f/N:F1}us(max {jM*f:F1})");
        }
        return resp;
    });
}
else
{
    app.MapPost("/fraud-score", (FraudRequest request) =>
    {
        Span<float> query = stackalloc float[Dataset.Dimensions];
        vectorizer.Vectorize(request, query);
        var score = scorer.Score(query);
        return Results.Json(new FraudResponse(score < 0.6f, score), AppJsonContext.Default.FraudResponse);
    });
}

app.Lifetime.ApplicationStarted.Register(() =>
{
    var simd = System.Runtime.Intrinsics.Vector256.IsHardwareAccelerated ? "AVX2"
             : System.Runtime.Intrinsics.Vector128.IsHardwareAccelerated ? "SSE-only (slow)"
             : "scalar (very slow)";
    var ivf = dataset.HasIvf ? $" IVF: {dataset.NumCells} cells." : "";
    Console.WriteLine($"Ready. Dataset: {dataset.Count:N0} vectors. Scorer: {scorerName}. SIMD: {simd}.{ivf}");
    // J11a: chmod the UDS so nginx (different user) can connect.
    var uds = Environment.GetEnvironmentVariable("UDS_PATH");
    if (!string.IsNullOrEmpty(uds) && File.Exists(uds))
    {
        try
        {
            File.SetUnixFileMode(uds,
                UnixFileMode.UserRead  | UnixFileMode.UserWrite  |
                UnixFileMode.GroupRead | UnixFileMode.GroupWrite |
                UnixFileMode.OtherRead | UnixFileMode.OtherWrite);
        }
        catch (Exception ex) { Console.Error.WriteLine($"chmod uds failed: {ex.Message}"); }
    }
});

app.Run();
