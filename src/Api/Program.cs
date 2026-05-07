using Rinha.Api;
using Microsoft.AspNetCore.Server.Kestrel.Core;
using System.IO.Pipelines;

// J19: pin ThreadPool to match cgroup CPU quota (0.45 cpu × 2 replicas).
// Hill-climbing can inject extra workers that fight over the fractional quota,
// adding context-switch overhead (~5-10us) directly to p99. With CPU-bound
// scoring and no blocking I/O, 1 worker + 1 IO thread is the right shape.
//
// L2-cascade follow-up: when a small fraction of requests run a 3ms IVF path
// while the bulk runs in ~10µs (cascade fast-path), 2 workers can both be
// occupied by IVF tail simultaneously and the fast-path queue piles up,
// inflating p99 dramatically. Env override TP_MAX_WORKERS lets us widen the
// pool when running with a mixed-latency handler.
var tpMin = int.TryParse(Environment.GetEnvironmentVariable("TP_MIN_WORKERS"), out var _tpMin) && _tpMin > 0 ? _tpMin : 1;
var tpMax = int.TryParse(Environment.GetEnvironmentVariable("TP_MAX_WORKERS"), out var _tpMax) && _tpMax > 0 ? _tpMax : 2;
var ioMin = int.TryParse(Environment.GetEnvironmentVariable("TP_MIN_IO"), out var _ioMin) && _ioMin > 0 ? _ioMin : 1;
var ioMax = int.TryParse(Environment.GetEnvironmentVariable("TP_MAX_IO"), out var _ioMax) && _ioMax > 0 ? _ioMax : 2;
ThreadPool.SetMinThreads(workerThreads: tpMin, completionPortThreads: ioMin);
ThreadPool.SetMaxThreads(workerThreads: tpMax, completionPortThreads: ioMax);

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
var vectorsQ16Path = Environment.GetEnvironmentVariable("VECTORS_Q16_PATH"); // optional (J25)
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
var dataset = Dataset.Open(vectorsPath, labelsPath, vectorsQ8Path, vectorsQ8SoaPath, vectorsQ16Path, ivfCentroidsPath, ivfOffsetsPath, ivfBboxMinPath, ivfBboxMaxPath, pqCodebooksPath, pqCodesPath, pqM, pqKsub);
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
                      $"jit={warmupIters} iters in {swJit.ElapsedMilliseconds}ms, " +
                      $"thp_advised={dataset.LastHugepageAdvisedBytes / (1024 * 1024)}MiB, " +
                      $"mlocked={dataset.LastMlockedBytes / (1024 * 1024)}MiB.");
}

var app = builder.Build();

// J20: LIFO concurrency limiter — admission control to defeat the "bench paradox".
// k6 ramping-arrival-rate floods VUs as soon as queue drains; serving "newest first"
// (LIFO) means in-progress requests finish under their deadline while old queued
// requests are dropped fast. PermitLimit matches CPU/threadpool budget per replica.
var lifoLimitEnv = Environment.GetEnvironmentVariable("LIFO_LIMIT");
if (!string.IsNullOrEmpty(lifoLimitEnv) && int.TryParse(lifoLimitEnv, out var lifoPermits) && lifoPermits > 0)
{
    var lifoQueue = int.TryParse(Environment.GetEnvironmentVariable("LIFO_QUEUE"), out var q) ? q : 32;
    var limiter = new System.Threading.RateLimiting.ConcurrencyLimiter(
        new System.Threading.RateLimiting.ConcurrencyLimiterOptions
        {
            PermitLimit = lifoPermits,
            QueueLimit = lifoQueue,
            QueueProcessingOrder = System.Threading.RateLimiting.QueueProcessingOrder.NewestFirst,
        });
    Console.WriteLine($"LIFO admission: permits={lifoPermits} queue={lifoQueue}");
    app.Use(async (ctx, next) =>
    {
        using var lease = await limiter.AcquireAsync(1, ctx.RequestAborted).ConfigureAwait(false);
        if (!lease.IsAcquired)
        {
            ctx.Response.StatusCode = 503;
            return;
        }
        await next(ctx).ConfigureAwait(false);
    });
}

app.MapGet("/ready", () => Results.Ok());

var profileEnv = Environment.GetEnvironmentVariable("PROFILE_TIMING");
var profile = profileEnv == "1" || profileEnv == "2";
var whatIfMode = profileEnv == "2";

// Initialise cell-visit counter when in what-if mode (one entry per cell).
if (whatIfMode && dataset.HasIvf)
{
    Rinha.Api.Scorers.IvfScorer.CellVisits = new int[dataset.NumCells];
}

// J24: hard-query dataset collector — opt-in via HARDQ_DUMP_PATH env.
var hqDumpPath = Environment.GetEnvironmentVariable("HARDQ_DUMP_PATH");
if (!string.IsNullOrEmpty(hqDumpPath))
{
    Rinha.Api.HardQueryDump.Open(hqDumpPath);
    Console.WriteLine($"HardQueryDump: writing to {hqDumpPath}");
}

// J24: hard-query predictor — when HARDQ_DEADLINE_US > 0, run a tiny DT classifier
// on the input vector; if predicted "hard" (full IVF scan likely), apply this
// per-call deadline to bound p99 on the worst tail.
int hardqDeadlineUs = 0;
{
    var s = Environment.GetEnvironmentVariable("HARDQ_DEADLINE_US");
    if (!string.IsNullOrEmpty(s) && int.TryParse(s, out var v) && v > 0) hardqDeadlineUs = v;
    if (hardqDeadlineUs > 0) Console.WriteLine($"HardQueryClassifier: enabled, deadline={hardqDeadlineUs}µs on hard");
}

// L4: adaptive nProbe — drop nProbe on classifier-predicted EASY queries
// (the bulk, ~95.7%) to save Q8 scan work; keep full nProbe on HARD queries.
// Disabled when 0 (use static IVF_NPROBE for everyone).
int hardqNProbeEasy = 0;
{
    var s = Environment.GetEnvironmentVariable("HARDQ_NPROBE_EASY");
    if (!string.IsNullOrEmpty(s) && int.TryParse(s, out var v) && v > 0) hardqNProbeEasy = v;
    if (hardqNProbeEasy > 0) Console.WriteLine($"HardQueryClassifier: nProbe={hardqNProbeEasy} on easy queries");
}

// L1: optional batched scoring — coordinator pairs concurrent requests within a short
// window and runs IvfScorer.ScoreBatch2 (single sweep, two queries). Opt-in: set
// IVF_BATCH_WAIT_US > 0 to enable. Requires scorer to be IvfScorer.
Rinha.Api.Scorers.IvfBatchCoordinator? batchCoord = null;
{
    var s = Environment.GetEnvironmentVariable("IVF_BATCH_WAIT_US");
    if (!string.IsNullOrEmpty(s) && int.TryParse(s, out var v) && v > 0
        && scorer is Rinha.Api.Scorers.IvfScorer ivfForBatch)
    {
        batchCoord = new Rinha.Api.Scorers.IvfBatchCoordinator(ivfForBatch, v);
        Console.WriteLine($"IvfBatchCoordinator: enabled, pair_wait={v}µs");
    }
}

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
    // Per-mode latency stats: index 0=full, 1=es1, 2=es2.
    var modeHist = new long[3, Buckets];
    var modeCount = new long[3];
    var modeSumUs = new long[3];
    var modeMaxUs = new long[3];
    var modeRowsSum = new long[3];
    var lockObj = new object();
    int Window = whatIfMode ? 1000 : 5000;
    long startupGen0 = GC.CollectionCount(0);
    long startupGen1 = GC.CollectionCount(1);
    long startupGen2 = GC.CollectionCount(2);
    long lastGen0 = startupGen0, lastGen1 = startupGen1, lastGen2 = startupGen2;
    long lastAlloc = GC.GetTotalAllocatedBytes(precise: false);

    // What-if aggregator (active only when whatIfMode). Per-pct counts of: queries that would
    // have passed the early-stop gate (margin AND unanimous), queries that would have been
    // unanimous regardless of margin, and slack-sum (only counted when slack is finite).
    int wiPctCount = Rinha.Api.Scorers.IvfScorer.WhatIfPcts.Length;
    var wiPass      = new long[wiPctCount];
    var wiUnanimous = new long[wiPctCount];
    var wiSlackSum  = new double[wiPctCount];
    var wiSlackCnt  = new long[wiPctCount];
    // "earliest pct that fires" histogram: index = pct slot, value = how many queries
    // would have early-stopped first at that pct. Queries that never fire land in the
    // 'never' bucket.
    var wiFirstFire = new long[wiPctCount + 1]; // last slot = 'never'
    app.MapPost("/fraud-score", (FraudRequest request) =>
    {
        long t0 = System.Diagnostics.Stopwatch.GetTimestamp();
        Span<float> query = stackalloc float[Dataset.Dimensions];
        vectorizer.Vectorize(request, query);
        long t1 = System.Diagnostics.Stopwatch.GetTimestamp();
        float score;
        if (whatIfMode && scorer is Rinha.Api.Scorers.IvfScorer ivfScorer)
            score = ivfScorer.ScoreWithWhatIf(query);
        else
            score = scorer.Score(query);
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
            // Per-mode telemetry.
            int mIdx = esMode == 0 ? 0 : (esMode == 1 ? 1 : 2);
            modeCount[mIdx]++;
            modeSumUs[mIdx] += sUs;
            if (sUs > modeMaxUs[mIdx]) modeMaxUs[mIdx] = sUs;
            modeRowsSum[mIdx] += rows;
            modeHist[mIdx, bucket]++;
            // Maintain top-3 by score time, capturing rows + mode of those slow queries.
            if (s > sTopA) { sTopC = sTopB; sTopB = sTopA; sTopA = s;
                             rowsTopC = rowsTopB; rowsTopB = rowsTopA; rowsTopA = rows;
                             modeTopC = modeTopB; modeTopB = modeTopA; modeTopA = esMode; }
            else if (s > sTopB) { sTopC = sTopB; sTopB = s;
                                  rowsTopC = rowsTopB; rowsTopB = rows;
                                  modeTopC = modeTopB; modeTopB = esMode; }
            else if (s > sTopC) { sTopC = s; rowsTopC = rows; modeTopC = esMode; }
            if (whatIfMode)
            {
                var passArr = Rinha.Api.Scorers.IvfScorer.LastWhatIfPass;
                var slkArr  = Rinha.Api.Scorers.IvfScorer.LastWhatIfSlack;
                var unaArr  = Rinha.Api.Scorers.IvfScorer.LastWhatIfUnanimous;
                if (passArr is not null && slkArr is not null && unaArr is not null)
                {
                    int firstFire = wiPctCount; // 'never' bucket
                    for (int p = 0; p < wiPctCount; p++)
                    {
                        if (passArr[p] != 0)
                        {
                            wiPass[p]++;
                            if (firstFire == wiPctCount) firstFire = p;
                        }
                        if (unaArr[p] != 0) wiUnanimous[p]++;
                        float sl = slkArr[p];
                        if (!float.IsNegativeInfinity(sl) && !float.IsNaN(sl))
                        {
                            wiSlackSum[p] += sl;
                            wiSlackCnt[p]++;
                        }
                    }
                    wiFirstFire[firstFire]++;
                }
            }
            if (n >= Window) flush = true;
        }
        if (flush)
        {
            long N, vS, sS, jS, vM, sM, jM, tA, tB, tC;
            long mF, mE1, mE2, rS, rM;
            int rTA, rTB, rTC, mTA, mTB, mTC;
            long[] h = new long[Buckets];
            long[] mC = new long[3], mSum = new long[3], mMax = new long[3], mRows = new long[3];
            long[,] mH = new long[3, Buckets];
            long[] wPass = null!, wUna = null!, wFire = null!, wSlkCnt = null!;
            double[] wSlkSum = null!;
            lock (lockObj)
            {
                N = n; vS = vSum; sS = sSum; jS = jSum; vM = vMax; sM = sMax; jM = jMax;
                tA = sTopA; tB = sTopB; tC = sTopC;
                mF = modeFull; mE1 = modeEarly1; mE2 = modeEarly2;
                rS = rowsSum; rM = rowsMax;
                rTA = rowsTopA; rTB = rowsTopB; rTC = rowsTopC;
                mTA = modeTopA; mTB = modeTopB; mTC = modeTopC;
                Array.Copy(hist, h, Buckets);
                for (int mi = 0; mi < 3; mi++)
                {
                    mC[mi] = modeCount[mi]; mSum[mi] = modeSumUs[mi];
                    mMax[mi] = modeMaxUs[mi]; mRows[mi] = modeRowsSum[mi];
                    for (int b = 0; b < Buckets; b++) mH[mi, b] = modeHist[mi, b];
                }
                Array.Clear(modeCount); Array.Clear(modeSumUs);
                Array.Clear(modeMaxUs); Array.Clear(modeRowsSum);
                Array.Clear(modeHist);
                if (whatIfMode)
                {
                    wPass = (long[])wiPass.Clone();
                    wUna  = (long[])wiUnanimous.Clone();
                    wFire = (long[])wiFirstFire.Clone();
                    wSlkSum = (double[])wiSlackSum.Clone();
                    wSlkCnt = (long[])wiSlackCnt.Clone();
                    Array.Clear(wiPass);
                    Array.Clear(wiUnanimous);
                    Array.Clear(wiFirstFire);
                    Array.Clear(wiSlackSum);
                    Array.Clear(wiSlackCnt);
                }
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

            // Per-mode latency: dump histogram + p50/p99 derived from buckets.
            string[] modeNames = { "full", "es1", "es2" };
            for (int mi = 0; mi < 3; mi++)
            {
                long mn = mC[mi];
                if (mn == 0) continue;
                long target50 = mn / 2, target99 = (mn * 99) / 100;
                long acc = 0; int p50b = -1, p99b = -1;
                for (int b = 0; b < Buckets; b++)
                {
                    acc += mH[mi, b];
                    if (p50b < 0 && acc >= target50) p50b = b;
                    if (p99b < 0 && acc >= target99) { p99b = b; break; }
                }
                long p50Lo = p50b < 0 ? 0 : (100L << p50b);
                long p99Lo = p99b < 0 ? 0 : (100L << p99b);
                var sbm = new System.Text.StringBuilder();
                for (int b = 0; b < Buckets; b++)
                {
                    sbm.Append(100L << b).Append(':').Append(mH[mi, b]).Append(' ');
                }
                Console.WriteLine(
                    $"  mode={modeNames[mi]} N={mn} mean={mSum[mi]/(double)mn:F0}us " +
                    $"p50≥{p50Lo}us p99≥{p99Lo}us max={mMax[mi]}us " +
                    $"rows-avg={mRows[mi]/mn:N0} hist[us:{sbm}]");
            }

            if (whatIfMode)
            {
                var pcts = Rinha.Api.Scorers.IvfScorer.WhatIfPcts;
                var sbWi = new System.Text.StringBuilder();
                sbWi.Append("[whatif N=").Append(N).Append("] pct gate-pass% unanimous% mean-slack first-fire%\n");
                for (int p = 0; p < pcts.Length; p++)
                {
                    double gatePct = N == 0 ? 0 : 100.0 * wPass[p] / N;
                    double unaPct  = N == 0 ? 0 : 100.0 * wUna[p] / N;
                    double meanSlk = wSlkCnt[p] == 0 ? 0 : wSlkSum[p] / wSlkCnt[p];
                    double firePct = N == 0 ? 0 : 100.0 * wFire[p] / N;
                    sbWi.Append($"  pct={pcts[p],3} {gatePct,7:F1}%  {unaPct,7:F1}%  {meanSlk,+9:F4}  {firePct,7:F1}%\n");
                }
                double neverPct = N == 0 ? 0 : 100.0 * wFire[pcts.Length] / N;
                sbWi.Append($"  pct=---     ---       ---       ---       {neverPct,7:F1}%  (never fires)\n");

                // Estimate score for each pct: approximate scan_us using mode breakdown of CURRENT
                // config (not pct-specific yet — refine in next iteration). For now, just note that
                // higher first-fire% at smaller pct = lower expected p99.
                Console.WriteLine(sbWi.ToString().TrimEnd());

                // Cell-visit hot/cold summary.
                var visits = Rinha.Api.Scorers.IvfScorer.CellVisits;
                if (visits is not null)
                {
                    long total = 0, max = 0, min = long.MaxValue, zero = 0;
                    for (int i = 0; i < visits.Length; i++)
                    {
                        long cv = visits[i];
                        total += cv;
                        if (cv > max) max = cv;
                        if (cv < min) min = cv;
                        if (cv == 0) zero++;
                    }
                    double avg = (double)total / Math.Max(1, visits.Length);
                    Console.WriteLine($"[whatif N={N}] cells visited: avg={avg:F0} min={min} max={max} unused-cells={zero}/{visits.Length}");
                }
            }
        }
        return resp;
    });
}
else if (fastJson)
{
    // J11c: bypass model binding entirely. Read raw body bytes via PipeReader,
    // parse straight into a Span<float>, write the response by hand.

    // Optional GC/alloc telemetry: ALLOC_TRACE=1 dumps gen0/1/2 + bytes per 5000 reqs.
    var allocTrace = Environment.GetEnvironmentVariable("ALLOC_TRACE") == "1";
    long _atN = 0;
    long _atG0 = GC.CollectionCount(0), _atG1 = GC.CollectionCount(1), _atG2 = GC.CollectionCount(2);
    long _atAlloc = GC.GetTotalAllocatedBytes(precise: false);
    var _atLock = new object();
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

        float score;
        // Cascade-rejected queries are inherently uncertain — apply the hard-query
        // deadline (same one HardQueryClassifier uses) to keep p99 bounded.
        if (hardqDeadlineUs > 0) Rinha.Api.Scorers.IvfScorer.CallDeadlineUs = hardqDeadlineUs;
        score = scorer.Score(query);
        Rinha.Api.Scorers.IvfScorer.CallDeadlineUs = 0;

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

        if (allocTrace)
        {
            bool dump = false;
            long g0=0, g1=0, g2=0, dAlloc=0, N=0;
            lock (_atLock)
            {
                _atN++;
                if (_atN >= 5000)
                {
                    long ng0 = GC.CollectionCount(0), ng1 = GC.CollectionCount(1), ng2 = GC.CollectionCount(2);
                    long nAlloc = GC.GetTotalAllocatedBytes(precise: false);
                    g0 = ng0 - _atG0; g1 = ng1 - _atG1; g2 = ng2 - _atG2;
                    dAlloc = nAlloc - _atAlloc;
                    N = _atN;
                    _atG0 = ng0; _atG1 = ng1; _atG2 = ng2; _atAlloc = nAlloc; _atN = 0;
                    dump = true;
                }
            }
            if (dump)
            {
                Console.WriteLine($"[alloc N={N}] gc[g0/g1/g2={g0}/{g1}/{g2}] alloc={dAlloc/1024}KB total={dAlloc} bytes_per_req={dAlloc/(double)N:F1}");
            }
        }
    });
}
else
{
    if (batchCoord is not null)
    {
        // L1 batched path: handler is async because we await the worker's TCS.
        app.MapPost("/fraud-score", async (FraudRequest request) =>
        {
            var qArr = new float[Dataset.Dimensions];
            vectorizer.Vectorize(request, qArr.AsSpan());
            float bs = await batchCoord.SubmitAsync(qArr).ConfigureAwait(false);
            return Results.Json(new FraudResponse(bs < 0.6f, bs), AppJsonContext.Default.FraudResponse);
        });
    }
    else
    {
        app.MapPost("/fraud-score", (FraudRequest request) =>
        {
            Span<float> query = stackalloc float[Dataset.Dimensions];
            vectorizer.Vectorize(request, query);
            float score;
            bool isHard = (hardqDeadlineUs > 0 || hardqNProbeEasy > 0)
                && Rinha.Api.Scorers.HardQueryClassifier.IsHard(query);
            int dl = isHard ? hardqDeadlineUs : (hardqNProbeEasy > 0 ? hardqDeadlineUs : 0);
            if (dl > 0) Rinha.Api.Scorers.IvfScorer.CallDeadlineUs = dl;
            if (!isHard && hardqNProbeEasy > 0)
            {
                Rinha.Api.Scorers.IvfScorer.CallNProbe = hardqNProbeEasy;
            }
            score = scorer.Score(query);
            Rinha.Api.Scorers.IvfScorer.CallDeadlineUs = 0;
            Rinha.Api.Scorers.IvfScorer.CallNProbe = 0;
            if (Rinha.Api.HardQueryDump.Enabled)
            {
                Rinha.Api.HardQueryDump.Append(query,
                    Rinha.Api.Scorers.IvfScorer.LastRowsScanned,
                    Rinha.Api.Scorers.IvfScorer.LastEarlyStopMode);
            }
            return Results.Json(new FraudResponse(score < 0.6f, score), AppJsonContext.Default.FraudResponse);
        });
    }
}

app.Lifetime.ApplicationStopping.Register(() => Rinha.Api.HardQueryDump.Close());

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
