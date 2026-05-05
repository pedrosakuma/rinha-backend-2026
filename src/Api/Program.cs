using Rinha.Api;
using Microsoft.AspNetCore.Server.Kestrel.Core;
using System.IO.Pipelines;

var builder = WebApplication.CreateSlimBuilder(args);

builder.Logging.ClearProviders();

builder.WebHost.ConfigureKestrel(options =>
{
    options.AddServerHeader = false;
    options.AllowSynchronousIO = false;
    // J4: tight limits — body is ~400 bytes; cap to skip large-body code paths.
    options.Limits.MaxRequestBodySize = 8 * 1024;
    options.Limits.MaxRequestHeadersTotalSize = 4 * 1024;
    options.Limits.MaxRequestLineSize = 1 * 1024;
    options.Limits.MaxConcurrentUpgradedConnections = 0;
    options.Limits.KeepAliveTimeout = TimeSpan.FromMinutes(5);
    options.Limits.RequestHeadersTimeout = TimeSpan.FromSeconds(5);

    var udsPath = Environment.GetEnvironmentVariable("UDS_PATH");
    if (!string.IsNullOrEmpty(udsPath))
    {
        // J11a: listen on a Unix Domain Socket so the LB upstream is local FS, not TCP loopback.
        // Saves ~30-50us per request (no TCP/IP stack, no port allocation, no Nagle delays).
        if (File.Exists(udsPath)) File.Delete(udsPath);
        options.ListenUnixSocket(udsPath, lo => lo.Protocols = HttpProtocols.Http1);
    }
    else
    {
        var port = int.Parse(Environment.GetEnvironmentVariable("PORT") ?? "9999");
        options.ListenAnyIP(port, lo => lo.Protocols = HttpProtocols.Http1);
    }
});

builder.Services.ConfigureHttpJsonOptions(o =>
{
    o.SerializerOptions.TypeInfoResolverChain.Insert(0, AppJsonContext.Default);
});

var vectorsPath = Environment.GetEnvironmentVariable("VECTORS_PATH") ?? "/data/references.bin";
var labelsPath = Environment.GetEnvironmentVariable("LABELS_PATH") ?? "/data/labels.bin";
var vectorsQ8Path = Environment.GetEnvironmentVariable("VECTORS_Q8_PATH"); // optional
var vectorsQ8SoaPath = Environment.GetEnvironmentVariable("VECTORS_Q8_SOA_PATH"); // optional (J11)
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
var dataset = Dataset.Open(vectorsPath, labelsPath, vectorsQ8Path, vectorsQ8SoaPath, ivfCentroidsPath, ivfOffsetsPath, ivfBboxMinPath, ivfBboxMaxPath, pqCodebooksPath, pqCodesPath, pqM, pqKsub);
var vectorizer = new Vectorizer(normalization, mccRisk);
var jsonVectorizer = new JsonVectorizer(normalization, mccRisk);
var fastJson = Environment.GetEnvironmentVariable("FAST_JSON") == "1";
var scorerName = Environment.GetEnvironmentVariable("SCORER") ?? "brute";
IFraudScorer scorer = ScorerFactory.Create(scorerName, dataset);

// Pre-touch all mmap pages and run a JIT warm-up so the first user requests
// don't pay page-fault or tier0->tier1 jit overhead. Disable with WARMUP=0.
if (Environment.GetEnvironmentVariable("WARMUP") != "0")
{
    var swPre = System.Diagnostics.Stopwatch.StartNew();
    long touched = dataset.Prefetch();
    swPre.Stop();
    var swJit = System.Diagnostics.Stopwatch.StartNew();
    int warmupIters = int.TryParse(Environment.GetEnvironmentVariable("WARMUP_ITERS"), out var _wi) && _wi > 0 ? _wi : 64;
    Span<float> warmQ = stackalloc float[Dataset.Dimensions];
    var rng = new Random(20260505);
    float warmSink = 0;
    for (int i = 0; i < warmupIters; i++)
    {
        for (int d = 0; d < Dataset.Dimensions; d++)
            warmQ[d] = (float)(rng.NextDouble() * 2 - 1);
        warmSink += scorer.Score(warmQ);
    }
    swJit.Stop();
    GC.KeepAlive(warmSink);
    Console.WriteLine($"Warm-up: prefetch={touched / (1024 * 1024)}MiB in {swPre.ElapsedMilliseconds}ms, " +
                      $"jit={warmupIters} iters in {swJit.ElapsedMilliseconds}ms.");
}

var app = builder.Build();

app.MapGet("/ready", () => Results.Ok());

var profile = Environment.GetEnvironmentVariable("PROFILE_TIMING") == "1";
if (profile)
{
    long n = 0, vSum = 0, sSum = 0, jSum = 0;
    long vMax = 0, sMax = 0, jMax = 0;
    long modeFull = 0, modeEarly1 = 0, modeEarly2 = 0;
    long rowsSum = 0, rowsMax = 0;
    // Score-time histogram: log-scale buckets (us). Bucket i covers [2^i * 100us, 2^(i+1) * 100us).
    // 12 buckets cover 100us..409.6ms.
    const int Buckets = 12;
    var hist = new long[Buckets];
    long sTopA = 0, sTopB = 0, sTopC = 0; // top-3 score-times in window (us ticks)
    int rowsTopA = 0, rowsTopB = 0, rowsTopC = 0;
    int modeTopA = 0, modeTopB = 0, modeTopC = 0;
    var lockObj = new object();
    const int Window = 5000;
    long startupGen0 = GC.CollectionCount(0);
    long startupGen1 = GC.CollectionCount(1);
    long startupGen2 = GC.CollectionCount(2);
    long lastGen0 = startupGen0, lastGen1 = startupGen1, lastGen2 = startupGen2;
    long lastAlloc = GC.GetTotalAllocatedBytes(precise: false);
    app.MapPost("/fraud-score", (FraudRequest request) =>
    {
        long t0 = System.Diagnostics.Stopwatch.GetTimestamp();
        Span<float> query = stackalloc float[Dataset.Dimensions];
        vectorizer.Vectorize(request, query);
        long t1 = System.Diagnostics.Stopwatch.GetTimestamp();
        var score = scorer.Score(query);
        long t2 = System.Diagnostics.Stopwatch.GetTimestamp();
        // Read scorer telemetry (thread-static, set by IvfScorer).
        int esMode = Rinha.Api.Scorers.IvfScorer.LastEarlyStopMode;
        int rows = Rinha.Api.Scorers.IvfScorer.LastRowsScanned;
        var resp = Results.Json(new FraudResponse(score < 0.6f, score), AppJsonContext.Default.FraudResponse);
        long t3 = System.Diagnostics.Stopwatch.GetTimestamp();

        long v = t1 - t0, s = t2 - t1, j = t3 - t2;
        // Convert score-ticks to us for histogram bucket selection.
        long sUs = s * 1_000_000 / System.Diagnostics.Stopwatch.Frequency;
        int bucket = sUs <= 0 ? 0 : System.Numerics.BitOperations.Log2((ulong)Math.Max(1, sUs / 100));
        if (bucket >= Buckets) bucket = Buckets - 1;

        bool flush = false;
        lock (lockObj)
        {
            n++;
            vSum += v; sSum += s; jSum += j;
            if (v > vMax) vMax = v;
            if (s > sMax) sMax = s;
            if (j > jMax) jMax = j;
            hist[bucket]++;
            rowsSum += rows;
            if (rows > rowsMax) rowsMax = rows;
            if (esMode == 0) modeFull++;
            else if (esMode == 1) modeEarly1++;
            else modeEarly2++;
            // Maintain top-3 by score time, capturing rows + mode of those slow queries.
            if (s > sTopA) { sTopC = sTopB; sTopB = sTopA; sTopA = s;
                             rowsTopC = rowsTopB; rowsTopB = rowsTopA; rowsTopA = rows;
                             modeTopC = modeTopB; modeTopB = modeTopA; modeTopA = esMode; }
            else if (s > sTopB) { sTopC = sTopB; sTopB = s;
                                  rowsTopC = rowsTopB; rowsTopB = rows;
                                  modeTopC = modeTopB; modeTopB = esMode; }
            else if (s > sTopC) { sTopC = s; rowsTopC = rows; modeTopC = esMode; }
            if (n >= Window) flush = true;
        }
        if (flush)
        {
            long N, vS, sS, jS, vM, sM, jM, tA, tB, tC;
            long mF, mE1, mE2, rS, rM;
            int rTA, rTB, rTC, mTA, mTB, mTC;
            long[] h = new long[Buckets];
            lock (lockObj)
            {
                N = n; vS = vSum; sS = sSum; jS = jSum; vM = vMax; sM = sMax; jM = jMax;
                tA = sTopA; tB = sTopB; tC = sTopC;
                mF = modeFull; mE1 = modeEarly1; mE2 = modeEarly2;
                rS = rowsSum; rM = rowsMax;
                rTA = rowsTopA; rTB = rowsTopB; rTC = rowsTopC;
                mTA = modeTopA; mTB = modeTopB; mTC = modeTopC;
                Array.Copy(hist, h, Buckets);
                n = 0; vSum = sSum = jSum = 0; vMax = sMax = jMax = 0;
                sTopA = sTopB = sTopC = 0;
                rowsSum = rowsMax = 0;
                modeFull = modeEarly1 = modeEarly2 = 0;
                rowsTopA = rowsTopB = rowsTopC = 0;
                modeTopA = modeTopB = modeTopC = 0;
                Array.Clear(hist, 0, Buckets);
            }
            // GC delta in window.
            long g0 = GC.CollectionCount(0), g1 = GC.CollectionCount(1), g2 = GC.CollectionCount(2);
            long alloc = GC.GetTotalAllocatedBytes(precise: false);
            long dG0 = g0 - lastGen0, dG1 = g1 - lastGen1, dG2 = g2 - lastGen2;
            long dAlloc = alloc - lastAlloc;
            lastGen0 = g0; lastGen1 = g1; lastGen2 = g2; lastAlloc = alloc;

            double f = 1e6 / System.Diagnostics.Stopwatch.Frequency;
            // Build histogram string: bucket boundaries in us.
            var sb = new System.Text.StringBuilder();
            for (int i = 0; i < Buckets; i++)
            {
                long lo = 100L << i;
                sb.Append(lo).Append(':').Append(h[i]).Append(' ');
            }
            Console.WriteLine(
                $"[timing N={N}] vec={vS*f/N:F1}us(max {vM*f:F1}) " +
                $"score={sS*f/N:F1}us(max {sM*f:F1}) " +
                $"json={jS*f/N:F1}us(max {jM*f:F1}) " +
                $"rows-avg={rS/N:N0}/max={rM:N0} " +
                $"mode[full={mF * 100 / N}% es1={mE1 * 100 / N}% es2={mE2 * 100 / N}%] " +
                $"top3=[{tA*f:F0}us r={rTA:N0} m={mTA}] [{tB*f:F0}us r={rTB:N0} m={mTB}] [{tC*f:F0}us r={rTC:N0} m={mTC}] " +
                $"gc[g0/g1/g2={dG0}/{dG1}/{dG2} alloc={dAlloc/1024}KB] " +
                $"score-hist[us:{sb}]");
        }
        return resp;
    });
}
else if (fastJson)
{
    // J11c: bypass model binding entirely. Read raw body bytes via PipeReader,
    // parse straight into a Span<float>, write the response by hand.
    app.MapPost("/fraud-score", async (HttpContext ctx) =>
    {
        var pipe = ctx.Request.BodyReader;
        var contentLength = ctx.Request.ContentLength ?? 0;
        // Pull until we have the full body or the pipe completes.
        ReadResult rr;
        while (true)
        {
            rr = await pipe.ReadAsync();
            if (rr.Buffer.Length >= contentLength || rr.IsCompleted) break;
            pipe.AdvanceTo(rr.Buffer.Start, rr.Buffer.End);
        }
        var buffer = rr.Buffer;

        Span<float> query = stackalloc float[Dataset.Dimensions];
        if (buffer.IsSingleSegment)
        {
            jsonVectorizer.VectorizeJson(buffer.FirstSpan, query);
        }
        else
        {
            // Multi-segment is rare for ~500B bodies on local TCP; copy to a stack scratch.
            int len = (int)buffer.Length;
            if (len > 8 * 1024) len = 8 * 1024;
            Span<byte> scratch = stackalloc byte[8 * 1024];
            var slice = buffer.Slice(0, len);
            int p = 0;
            foreach (var seg in slice)
            {
                seg.Span.CopyTo(scratch[p..]);
                p += seg.Length;
            }
            jsonVectorizer.VectorizeJson(scratch[..len], query);
        }
        pipe.AdvanceTo(buffer.End);

        var score = scorer.Score(query);

        // Hand-written response: "{\"approved\":true|false,\"fraud_score\":<float>}"
        // Allocation-free path through Response.BodyWriter (PipeWriter) — get a memory
        // span, write directly, Advance, FlushAsync.
        ctx.Response.StatusCode = 200;
        ctx.Response.ContentType = "application/json";

        var bw = ctx.Response.BodyWriter;
        var outMem = bw.GetMemory(64);
        var outSpan = outMem.Span;
        int o = 0;
        ReadOnlySpan<byte> p1True = "{\"approved\":true,\"fraud_score\":"u8;
        ReadOnlySpan<byte> p1False = "{\"approved\":false,\"fraud_score\":"u8;
        var prefix = score < 0.6f ? p1True : p1False;
        prefix.CopyTo(outSpan);
        o += prefix.Length;
        if (System.Buffers.Text.Utf8Formatter.TryFormat(score, outSpan[o..], out var written))
            o += written;
        outSpan[o++] = (byte)'}';
        bw.Advance(o);
        await bw.FlushAsync();
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
