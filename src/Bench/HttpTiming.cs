using System.Diagnostics;
using System.Net.Http;
using System.Text;
using System.Text.Json;

namespace Rinha.Bench;

/// <summary>
/// Measures end-to-end HTTP latency of POST /fraud-score against a running API,
/// to compare with the in-process Replay scorer-only timings. Delta = HTTP +
/// socket + Kestrel + JSON-parse + serialize overhead.
///
/// Usage:
///   Rinha.Bench --http --url=http://localhost:9999/fraud-score
///       [--test-data=/tmp/rinha-eval/test/test-data.json]
///       [--limit=20000] [--concurrency=1] [--warmup=1000]
/// </summary>
public static class HttpTiming
{
    public static int Run(string[] args)
    {
        string url = "http://localhost:9999/fraud-score";
        string testData = "/tmp/rinha-eval/test/test-data.json";
        int limit = 20000;
        int concurrency = 1;
        int warmup = 1000;
        string? udsPath = null;

        foreach (var a in args)
        {
            if (a.StartsWith("--url=")) url = a.Substring(6);
            else if (a.StartsWith("--test-data=")) testData = a.Substring(12);
            else if (a.StartsWith("--limit=")) limit = int.Parse(a.Substring(8));
            else if (a.StartsWith("--concurrency=")) concurrency = int.Parse(a.Substring(14));
            else if (a.StartsWith("--warmup=")) warmup = int.Parse(a.Substring(9));
            else if (a.StartsWith("--uds=")) udsPath = a.Substring(6);
        }

        Console.WriteLine($"[http] url={url} uds={udsPath ?? "-"} limit={limit} concurrency={concurrency} warmup={warmup}");

        // Load raw request bodies from test-data.json
        var bytes = File.ReadAllBytes(testData);
        using var doc = JsonDocument.Parse(bytes);
        var entries = doc.RootElement.GetProperty("entries");
        var bodies = new List<byte[]>(Math.Min(entries.GetArrayLength(), limit));
        int n = 0;
        foreach (var entry in entries.EnumerateArray())
        {
            if (n >= limit) break;
            var req = entry.GetProperty("request");
            bodies.Add(Encoding.UTF8.GetBytes(req.GetRawText()));
            n++;
        }
        Console.WriteLine($"[http] loaded {bodies.Count} bodies");

        // SocketsHttpHandler: HTTP/1.1, keep-alive, one pooled connection per worker.
        var handler = new SocketsHttpHandler
        {
            MaxConnectionsPerServer = concurrency,
            PooledConnectionLifetime = TimeSpan.FromMinutes(10),
            PooledConnectionIdleTimeout = TimeSpan.FromMinutes(10),
            EnableMultipleHttp2Connections = false,
            UseProxy = false,
        };
        if (udsPath is not null)
        {
            // Dial Unix domain socket regardless of URL host (URL still parsed for path).
            handler.ConnectCallback = async (ctx, ct) =>
            {
                var sock = new System.Net.Sockets.Socket(System.Net.Sockets.AddressFamily.Unix,
                    System.Net.Sockets.SocketType.Stream, System.Net.Sockets.ProtocolType.Unspecified);
                await sock.ConnectAsync(new System.Net.Sockets.UnixDomainSocketEndPoint(udsPath), ct).ConfigureAwait(false);
                sock.NoDelay = true;
                return new System.Net.Sockets.NetworkStream(sock, ownsSocket: true);
            };
        }
        var client = new HttpClient(handler) { Timeout = TimeSpan.FromSeconds(30) };
        client.DefaultRequestHeaders.ConnectionClose = false;
        client.DefaultRequestVersion = System.Net.HttpVersion.Version11;
        client.DefaultVersionPolicy = HttpVersionPolicy.RequestVersionOrLower;

        // Warmup
        Console.Write("[http] warmup... ");
        for (int i = 0; i < warmup; i++)
        {
            int idx = i % bodies.Count;
            var req = NewReq(url, bodies[idx]);
            using var resp = client.Send(req, HttpCompletionOption.ResponseContentRead);
            resp.Content.ReadAsByteArrayAsync().GetAwaiter().GetResult();
        }
        Console.WriteLine("done");

        // Sequential per-worker timing.
        long[] perReq = new long[bodies.Count * 1]; // single pass
        var sw = Stopwatch.StartNew();

        if (concurrency == 1)
        {
            for (int i = 0; i < bodies.Count; i++)
            {
                var t0 = Stopwatch.GetTimestamp();
                var req = NewReq(url, bodies[i]);
                using var resp = client.Send(req, HttpCompletionOption.ResponseContentRead);
                resp.Content.ReadAsByteArrayAsync().GetAwaiter().GetResult();
                perReq[i] = Stopwatch.GetTimestamp() - t0;
            }
        }
        else
        {
            int idx = 0;
            var threads = new Thread[concurrency];
            int total = bodies.Count;
            for (int w = 0; w < concurrency; w++)
            {
                threads[w] = new Thread(() =>
                {
                    while (true)
                    {
                        int i = Interlocked.Increment(ref idx) - 1;
                        if (i >= total) return;
                        var t0 = Stopwatch.GetTimestamp();
                        var req = NewReq(url, bodies[i]);
                        using var resp = client.Send(req, HttpCompletionOption.ResponseContentRead);
                        resp.Content.ReadAsByteArrayAsync().GetAwaiter().GetResult();
                        perReq[i] = Stopwatch.GetTimestamp() - t0;
                    }
                });
                threads[w].IsBackground = true;
                threads[w].Start();
            }
            foreach (var t in threads) t.Join();
        }

        sw.Stop();

        Array.Sort(perReq);
        double tickToUs = 1_000_000.0 / Stopwatch.Frequency;
        double p50 = perReq[(int)(perReq.Length * 0.50)] * tickToUs;
        double p90 = perReq[(int)(perReq.Length * 0.90)] * tickToUs;
        double p99 = perReq[(int)(perReq.Length * 0.99)] * tickToUs;
        double p999 = perReq[(int)(perReq.Length * 0.999)] * tickToUs;
        double pmax = perReq[^1] * tickToUs;
        double rps = bodies.Count * 1000.0 / sw.ElapsedMilliseconds;

        Console.WriteLine($"[http] {bodies.Count} reqs in {sw.ElapsedMilliseconds}ms => {rps:F0} rps (c={concurrency})");
        Console.WriteLine($"[http] p50={p50:F1}us p90={p90:F1}us p99={p99:F1}us p999={p999:F1}us max={pmax:F1}us");

        return 0;
    }

    private static HttpRequestMessage NewReq(string url, byte[] body)
    {
        var req = new HttpRequestMessage(HttpMethod.Post, url)
        {
            Content = new ByteArrayContent(body),
        };
        req.Content.Headers.ContentType = new System.Net.Http.Headers.MediaTypeHeaderValue("application/json");
        return req;
    }
}
