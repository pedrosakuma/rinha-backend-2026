using Rinha.Api;

var builder = WebApplication.CreateSlimBuilder(args);

builder.Logging.ClearProviders();
builder.WebHost.ConfigureKestrel(options =>
{
    options.AddServerHeader = false;
    options.AllowSynchronousIO = false;
    var port = int.Parse(Environment.GetEnvironmentVariable("PORT") ?? "9999");
    options.ListenAnyIP(port);
});

builder.Services.ConfigureHttpJsonOptions(o =>
{
    o.SerializerOptions.TypeInfoResolverChain.Insert(0, AppJsonContext.Default);
});

var vectorsPath = Environment.GetEnvironmentVariable("VECTORS_PATH") ?? "/data/references.bin";
var labelsPath = Environment.GetEnvironmentVariable("LABELS_PATH") ?? "/data/labels.bin";
var vectorsQ8Path = Environment.GetEnvironmentVariable("VECTORS_Q8_PATH"); // optional
var mccRiskPath = Environment.GetEnvironmentVariable("MCC_RISK_PATH") ?? "/app/resources/mcc_risk.json";
var normalizationPath = Environment.GetEnvironmentVariable("NORMALIZATION_PATH") ?? "/app/resources/normalization.json";

var normalization = NormalizationConstants.Load(normalizationPath);
var mccRisk = MccRiskTable.Load(mccRiskPath);
var dataset = Dataset.Open(vectorsPath, labelsPath, vectorsQ8Path);
var vectorizer = new Vectorizer(normalization, mccRisk);
var scorerName = Environment.GetEnvironmentVariable("SCORER") ?? "brute";
IFraudScorer scorer = ScorerFactory.Create(scorerName, dataset);

var app = builder.Build();

app.MapGet("/ready", () => Results.Ok());

app.MapPost("/fraud-score", (FraudRequest request) =>
{
    Span<float> query = stackalloc float[Dataset.Dimensions];
    vectorizer.Vectorize(request, query);
    var score = scorer.Score(query);
    return Results.Json(new FraudResponse(score < 0.6f, score), AppJsonContext.Default.FraudResponse);
});

app.Lifetime.ApplicationStarted.Register(() =>
{
    var simd = System.Runtime.Intrinsics.Vector256.IsHardwareAccelerated ? "AVX2"
             : System.Runtime.Intrinsics.Vector128.IsHardwareAccelerated ? "SSE-only (slow)"
             : "scalar (very slow)";
    Console.WriteLine($"Ready. Dataset: {dataset.Count:N0} vectors. Scorer: {scorerName}. SIMD: {simd}.");
});

app.Run();
