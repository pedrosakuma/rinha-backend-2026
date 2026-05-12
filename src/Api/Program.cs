using Rinha.Api;
using Microsoft.AspNetCore.Server.Kestrel.Core;
using Kestrel.Transport.IoUring;
using System.Buffers;
using System.IO.Pipelines;
using System.Runtime;

// Workstation+Concurrent GC defaults to Interactive, which targets pause time but still
// permits gen2 STW pauses that can blow past 30ms under steady allocation.
// SustainedLowLatency tells the GC to favor latency over memory by suppressing gen2
// collections (allocations grow until the runtime hits memory pressure). Safe here
// because the hot path (scoring) is largely zero-alloc.
GCSettings.LatencyMode = GCLatencyMode.SustainedLowLatency;

// Pin ThreadPool to match cgroup CPU quota. Hill-climbing can inject extra workers
// that fight over the fractional quota, adding context-switch overhead (~5-10us)
// directly to p99. With CPU-bound scoring and no blocking I/O, 1 worker + 1 IO
// thread is the right shape. TP_MAX_WORKERS lets us widen the pool when running
// with a mixed-latency handler.
var tpMin = int.TryParse(Environment.GetEnvironmentVariable("TP_MIN_WORKERS"), out var _tpMin) && _tpMin > 0 ? _tpMin : 1;
var tpMax = int.TryParse(Environment.GetEnvironmentVariable("TP_MAX_WORKERS"), out var _tpMax) && _tpMax > 0 ? _tpMax : 2;
ThreadPool.SetMinThreads(workerThreads: tpMin, completionPortThreads: 1);
ThreadPool.SetMaxThreads(workerThreads: tpMax, completionPortThreads: 2);

var builder = WebApplication.CreateSlimBuilder(args);

builder.Logging.ClearProviders();

// IO_URING_ENABLED=1: replace Kestrel's default Sockets transport with io_uring.
// Linux 5.1+ required; auto-falls back to sockets on unsupported systems.
if (Environment.GetEnvironmentVariable("IO_URING_ENABLED") == "1")
{
    builder.WebHost.UseIoUring(opts =>
    {
        var ringSize = int.TryParse(Environment.GetEnvironmentVariable("IO_URING_SIZE"), out var rs) && rs > 0 ? rs : 256;
        var maxConn = int.TryParse(Environment.GetEnvironmentVariable("IO_URING_MAX_CONN"), out var mc) && mc > 0 ? mc : 1024;
        opts.RingSize = ringSize;
        opts.MaxConnections = maxConn;
    });
}

builder.WebHost.ConfigureKestrel(options =>
{
    options.AddServerHeader = false;
    options.AllowSynchronousIO = false;
    // Body is ~400 bytes; cap to skip large-body code paths.
    options.Limits.MaxRequestBodySize = 8 * 1024;
    options.Limits.MaxRequestHeadersTotalSize = 4 * 1024;
    options.Limits.MaxRequestLineSize = 1 * 1024;
    options.Limits.MaxConcurrentUpgradedConnections = 0;
    options.Limits.KeepAliveTimeout = TimeSpan.FromMinutes(5);
    options.Limits.RequestHeadersTimeout = TimeSpan.FromSeconds(5);

    var udsPath = Environment.GetEnvironmentVariable("UDS_PATH");
    if (!string.IsNullOrEmpty(udsPath))
    {
        // Listen on a Unix Domain Socket so the LB upstream is local FS, not TCP loopback.
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
var vectorsQ8SoaPath = Environment.GetEnvironmentVariable("VECTORS_Q8_SOA_PATH"); // optional
var vectorsQ16Path = Environment.GetEnvironmentVariable("VECTORS_Q16_PATH"); // optional
var vectorsQ16SoaPath = Environment.GetEnvironmentVariable("VECTORS_Q16_SOA_PATH"); // optional (pre-transposed SoA for brute)
var vectorsQ16BlockedPath = Environment.GetEnvironmentVariable("VECTORS_Q16_BLOCKED_PATH"); // optional (block-SoA for ivf-blocked)
var ivfBlockOffsetsPath = Environment.GetEnvironmentVariable("IVF_BLOCK_OFFSETS_PATH"); // optional (block prefix sums)
var ivfCentroidsPath = Environment.GetEnvironmentVariable("IVF_CENTROIDS_PATH");
var ivfOffsetsPath = Environment.GetEnvironmentVariable("IVF_OFFSETS_PATH");
var ivfBboxMinPath = Environment.GetEnvironmentVariable("IVF_BBOX_MIN_PATH");
var ivfBboxMaxPath = Environment.GetEnvironmentVariable("IVF_BBOX_MAX_PATH");
var mccRiskPath = Environment.GetEnvironmentVariable("MCC_RISK_PATH") ?? "/app/resources/mcc_risk.json";
var normalizationPath = Environment.GetEnvironmentVariable("NORMALIZATION_PATH") ?? "/app/resources/normalization.json";

var normalization = NormalizationConstants.Load(normalizationPath);
var mccRisk = MccRiskTable.Load(mccRiskPath);
var dataset = Dataset.Open(vectorsPath, labelsPath, vectorsQ8Path, vectorsQ8SoaPath, vectorsQ16Path, vectorsQ16SoaPath, ivfCentroidsPath, ivfOffsetsPath, ivfBboxMinPath, ivfBboxMaxPath, vectorsQ16BlockedPath, ivfBlockOffsetsPath);
var vectorizer = new Vectorizer(normalization, mccRisk);
var jsonVectorizer = new JsonVectorizer(normalization, mccRisk);
var scorerName = Environment.GetEnvironmentVariable("SCORER") ?? "brute";
IFraudScorer scorer = ScorerFactory.Create(scorerName, dataset);
// Q16 capability: if the scorer accepts int16 queries directly, the hot path can
// vectorize straight to short and skip the round+cast inside the scorer.
IQ16FraudScorer? q16Scorer = scorer as IQ16FraudScorer;

var selectiveCascade = SelectiveDecisionCascade.Build(dataset, ResolveResourcePath("selective_decision_tables.json"));

// Pre-touch all mmap pages and run a JIT warm-up so the first user requests
// don't pay page-fault or tier0->tier1 jit overhead. Disable with WARMUP=0.
if (Environment.GetEnvironmentVariable("WARMUP") != "0")
{
    var swPre = System.Diagnostics.Stopwatch.StartNew();
    long touched = dataset.Prefetch(scorerName);
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
                      $"jit={warmupIters} iters in {swJit.ElapsedMilliseconds}ms, " +
                      $"thp_advised={dataset.LastHugepageAdvisedBytes / (1024 * 1024)}MiB, " +
                      $"mlocked={dataset.LastMlockedBytes / (1024 * 1024)}MiB.");
}

// NO_GC_REGION=<megabytes>: after warmup, enter a no-GC region so the GC does not
// interrupt requests during the eval window. Only safe with RAW_HTTP=1 (zero-alloc
// hot path). With Kestrel (RAW_HTTP=0) the async machinery allocates ~800B/request
// (~43 MB over 120s / 54k requests), which would exceed any practical budget and
// silently fall back to normal GC anyway.
// Example: NO_GC_REGION=20  (20 MiB is ample for the RAW_HTTP zero-alloc path)
if (int.TryParse(Environment.GetEnvironmentVariable("NO_GC_REGION"), out int noGcMb) && noGcMb > 0)
{
    GC.Collect(2, GCCollectionMode.Forced, blocking: true, compacting: true);
    long budget = (long)noGcMb * 1024 * 1024;
    bool entered = GC.TryStartNoGCRegion(budget, budget, induceCollection: false);
    Console.WriteLine(entered
        ? $"GC: no-GC region entered ({noGcMb} MiB SOH + {noGcMb} MiB LOH). Effective only with zero-alloc hot path (RAW_HTTP=1)."
        : $"GC: TryStartNoGCRegion({noGcMb} MiB) failed — insufficient free heap; running under SustainedLowLatency instead.");
}

var app = builder.Build();

// RAW_HTTP=1: skip Kestrel/ASP.NET entirely. RawHttpServer takes over the same
// UDS path Kestrel would have listened on, with manual HTTP/1.1 parsing,
// pre-built response buffers and the IFraudScorer.ScoreCount integer fast path.
// Built specifically to close the Kestrel-overhead gap to itagyba (#2781 = 5769 / p99 1.70ms).
if (Environment.GetEnvironmentVariable("RAW_HTTP") == "1")
{
    var rawUds = Environment.GetEnvironmentVariable("UDS_PATH");
    if (string.IsNullOrEmpty(rawUds))
        throw new InvalidOperationException("RAW_HTTP=1 requires UDS_PATH to be set.");

    // chmod is the LB-visible affordance; do it before Accept().
    Action chmodUds = () =>
    {
        try
        {
            File.SetUnixFileMode(rawUds,
                UnixFileMode.UserRead  | UnixFileMode.UserWrite  |
                UnixFileMode.GroupRead | UnixFileMode.GroupWrite |
                UnixFileMode.OtherRead | UnixFileMode.OtherWrite);
        }
        catch (Exception ex) { Console.Error.WriteLine($"chmod uds failed: {ex.Message}"); }
    };
    // RawHttpServer.Run binds, then we chmod from a side thread once the file exists.
    var chmodThread = new Thread(() =>
    {
        for (int i = 0; i < 50; i++)
        {
            if (File.Exists(rawUds)) { chmodUds(); return; }
            Thread.Sleep(20);
        }
    }) { IsBackground = true };
    chmodThread.Start();

    var simd = System.Runtime.Intrinsics.Vector256.IsHardwareAccelerated ? "AVX2"
             : System.Runtime.Intrinsics.Vector128.IsHardwareAccelerated ? "SSE-only (slow)" : "scalar";
    var ivf = dataset.HasIvf ? $" IVF: {dataset.NumCells} cells." : "";
    Console.WriteLine($"Ready (raw-http). Dataset: {dataset.Count:N0} vectors. Scorer: {scorerName}. SIMD: {simd}.{ivf}");

    Rinha.Api.RawHttpServer.Run(rawUds, scorer, jsonVectorizer, selectiveCascade);
    return; // never reached, but lets the compiler see the method ends.
}

// Use legacy path-branching (UseExtensions.Map, NOT IEndpointRouteBuilder.Map):
// terminates the pipeline by path prefix without invoking EndpointRoutingMiddleware
// or EndpointMiddleware. Profile (wave 22) showed those + RhpInterfaceDispatch1
// at ~1.5% combined CPU; this branch dispatches directly.
app.Map("/ready", branch => branch.Run(ctx =>
{
    ctx.Response.StatusCode = 200;
    return Task.CompletedTask;
}));

// Hot path: bypass model binding entirely. Read raw body bytes via PipeReader,
// parse straight into a Span<float>/Span<short>, then write the pre-baked response
// directly to BodyWriter (PipeWriter) so Kestrel emits status+headers+body in one
// chunk (one sendmsg). Combined with TCP_NODELAY this avoids both Nagle stalls
// and split sends.
//
// No `async` on the outer handler — for ~500B bodies on UDS the PipeReader already
// has data when the handler is invoked, so ReadAsync completes synchronously.
// Keeping the hot path free of async avoids per-request state machine allocations.
// The rare case (multi-read or async flush) falls through to an async local function.
app.Map("/fraud-score", branch => branch.Run(ctx =>
{
    var pipe = ctx.Request.BodyReader;
    var contentLength = ctx.Request.ContentLength ?? 0;

    // Fast path: body is already buffered in the pipe (single synchronous read).
    var readVt = pipe.ReadAsync();
    if (!readVt.IsCompletedSuccessfully)
        return FraudScoreSlowAsync(ctx, pipe, contentLength);

    var rr = readVt.Result;
    if (rr.Buffer.Length < contentLength && !rr.IsCompleted)
    {
        pipe.AdvanceTo(rr.Buffer.Start, rr.Buffer.End);
        return FraudScoreSlowAsync(ctx, pipe, contentLength);
    }

    // Zero-alloc synchronous hot path — no state machine, no heap allocations.
    var buffer = rr.Buffer;

    Span<float> query = stackalloc float[Dataset.Dimensions];
    Span<short> queryQ16 = stackalloc short[Dataset.Dimensions];
    bool useQ16 = q16Scorer is not null;
    // Wave 8: fast-path needs the float vector (edges are in float space). We always
    // produce both float + Q16 from a single parse; the cost over Q16-only is one extra
    // multiply per dim (~10ns).
    bool needFloat = !useQ16 || selectiveCascade.IsEnabled;
    if (buffer.IsSingleSegment)
    {
        if (needFloat) jsonVectorizer.VectorizeJson(buffer.FirstSpan, query, queryQ16);
        else           jsonVectorizer.VectorizeJsonQ16(buffer.FirstSpan, queryQ16);
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
        if (needFloat) jsonVectorizer.VectorizeJson(scratch[..len], query, queryQ16);
        else           jsonVectorizer.VectorizeJsonQ16(scratch[..len], queryQ16);
    }
    pipe.AdvanceTo(buffer.End);

    int idx;
    byte selective = selectiveCascade.IsEnabled
        ? selectiveCascade.TryLookup(query)
        : SelectiveDecisionCascade.ResultUndecided;
    if (selective != SelectiveDecisionCascade.ResultUndecided)
    {
        idx = selective;
    }
    else
    {
        float score = useQ16 ? q16Scorer!.ScoreQ16(queryQ16) : scorer.Score(query);
        idx = PrecomputedFraudResponse.ScoreToIndex(score);
    }

    // Pre-baked response: write directly to PipeWriter so Kestrel emits
    // status+headers+body in a single sendmsg.
    var body = PrecomputedFraudResponse.BodyForIndex(idx).Span;
    var resp = ctx.Response;
    resp.StatusCode = 200;
    resp.ContentType = "application/json";
    resp.ContentLength = body.Length;
    // Suppress Kestrel's per-request Date header (~25B + cost of formatting).
    // Setting an empty value keeps Kestrel from running its DateHeaderValueManager
    // for this response. Safe for our internal-only HTTP/1.1 + LB topology.
    resp.Headers.Date = default;
    var writer = resp.BodyWriter;
    writer.Write(body);
    var ft = writer.FlushAsync();
    return ft.IsCompletedSuccessfully ? Task.CompletedTask : ft.AsTask();

    // Slow path: body spans multiple pipe segments or ReadAsync didn't complete
    // synchronously. Allocates a state machine, but this is rare in practice.
    async Task FraudScoreSlowAsync(HttpContext slowCtx, PipeReader slowPipe, long slowContentLen)
    {
        ReadResult srr;
        while (true)
        {
            srr = await slowPipe.ReadAsync();
            if (srr.Buffer.Length >= slowContentLen || srr.IsCompleted) break;
            slowPipe.AdvanceTo(srr.Buffer.Start, srr.Buffer.End);
        }
        var sbuffer = srr.Buffer;

        // stackalloc is valid here: spans are created after the last ReadAsync await
        // and consumed before the FlushAsync await, so they don't cross a suspension.
        Span<float> squery = stackalloc float[Dataset.Dimensions];
        Span<short> squeryQ16 = stackalloc short[Dataset.Dimensions];
        bool suseQ16 = q16Scorer is not null;
        bool sneedFloat = !suseQ16 || selectiveCascade.IsEnabled;
        if (sbuffer.IsSingleSegment)
        {
            if (sneedFloat) jsonVectorizer.VectorizeJson(sbuffer.FirstSpan, squery, squeryQ16);
            else            jsonVectorizer.VectorizeJsonQ16(sbuffer.FirstSpan, squeryQ16);
        }
        else
        {
            int slen = (int)sbuffer.Length;
            if (slen > 8 * 1024) slen = 8 * 1024;
            Span<byte> sscratch = stackalloc byte[8 * 1024];
            var sslice = sbuffer.Slice(0, slen);
            int sp = 0;
            foreach (var seg in sslice) { seg.Span.CopyTo(sscratch[sp..]); sp += seg.Length; }
            if (sneedFloat) jsonVectorizer.VectorizeJson(sscratch[..slen], squery, squeryQ16);
            else            jsonVectorizer.VectorizeJsonQ16(sscratch[..slen], squeryQ16);
        }
        slowPipe.AdvanceTo(sbuffer.End);

        int sidx;
        byte sselective = selectiveCascade.IsEnabled
            ? selectiveCascade.TryLookup(squery)
            : SelectiveDecisionCascade.ResultUndecided;
        if (sselective != SelectiveDecisionCascade.ResultUndecided)
            sidx = sselective;
        else
        {
            float sscore = suseQ16 ? q16Scorer!.ScoreQ16(squeryQ16) : scorer.Score(squery);
            sidx = PrecomputedFraudResponse.ScoreToIndex(sscore);
        }

        var sbody = PrecomputedFraudResponse.BodyForIndex(sidx).Span;
        var sresp = slowCtx.Response;
        sresp.StatusCode = 200;
        sresp.ContentType = "application/json";
        sresp.ContentLength = sbody.Length;
        sresp.Headers.Date = default;
        var swriter = sresp.BodyWriter;
        swriter.Write(sbody);
        await swriter.FlushAsync();
    }
}));

app.Lifetime.ApplicationStarted.Register(() =>
{
    var simd = System.Runtime.Intrinsics.Vector256.IsHardwareAccelerated ? "AVX2"
             : System.Runtime.Intrinsics.Vector128.IsHardwareAccelerated ? "SSE-only (slow)"
             : "scalar (very slow)";
    var ivf = dataset.HasIvf ? $" IVF: {dataset.NumCells} cells." : "";
    Console.WriteLine($"Ready. Dataset: {dataset.Count:N0} vectors. Scorer: {scorerName}. SIMD: {simd}.{ivf}");
    // chmod the UDS so nginx (different user) can connect.
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

static string ResolveResourcePath(string fileName)
{
    string path = Path.Combine(AppContext.BaseDirectory, "resources", fileName);
    if (File.Exists(path)) return path;

    string? root = AppContext.BaseDirectory;
    while (root is not null && !File.Exists(Path.Combine(root, "Rinha.slnx")))
        root = Path.GetDirectoryName(root);
    return root is null ? path : Path.Combine(root, "resources", fileName);
}
