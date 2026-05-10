using System.Net.Sockets;
using System.Runtime.CompilerServices;

namespace Rinha.Api;

/// <summary>
/// Minimal HTTP/1.1 server hand-rolled for the Rinha hot path. Replaces Kestrel
/// when <c>RAW_HTTP=1</c>, listening on the same UDS path Kestrel would use.
///
/// Design choices (vs the simpler reference port):
///   * Skip ASP.NET entirely — no DI, no middleware, no HttpContext, no async state machines.
///   * <see cref="IFraudScorer.ScoreCount"/> integer fast path — eliminates the
///     <c>fraudCount → fraudCount/5f → MathF.Round(score*5f)</c> round-trip Kestrel forces.
///   * Pre-built <em>full</em> response buffers (status line + headers + body) — single
///     <see cref="Socket.Send(ReadOnlySpan{byte})"/> per request.
///   * Stack-allocated 4 KiB recv buffer — no <see cref="System.Buffers.ArrayPool{T}"/> ceremony.
///   * One dedicated thread per accepted connection (long-running keep-alive). With
///     CFS pinning to one logical CPU per replica, scheduler cost stays tiny;
///     k6 reuses connections so threads are typically a handful.
///   * Branch-light dispatch — happy path is a single <see cref="ReadOnlySpan{T}.StartsWith"/>
///     check on <c>"POST /fraud-score "</c>.
/// </summary>
internal static class RawHttpServer
{
    private const int RecvBufferSize = 4096;
    private const int Backlog = 512;

    private static readonly byte[] ReadyResponse =
        "HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: 2\r\nConnection: keep-alive\r\n\r\nOK"u8.ToArray();
    private static readonly byte[] NotFoundResponse =
        "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: keep-alive\r\n\r\n"u8.ToArray();
    private static readonly byte[] BadRequestResponse =
        "HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"u8.ToArray();
    private static readonly byte[] DefaultResponse = BuildFraudResponse(0);
    private static readonly byte[][] FraudResponses = new[]
    {
        BuildFraudResponse(0), BuildFraudResponse(1), BuildFraudResponse(2),
        BuildFraudResponse(3), BuildFraudResponse(4), BuildFraudResponse(5),
    };

    private static byte[] BuildFraudResponse(int fraudCount)
    {
        var body = PrecomputedFraudResponse.BodyFor(fraudCount);
        var prefix = $"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {body.Length}\r\nConnection: keep-alive\r\n\r\n";
        var prefixBytes = System.Text.Encoding.ASCII.GetBytes(prefix);
        var combined = new byte[prefixBytes.Length + body.Length];
        prefixBytes.CopyTo(combined, 0);
        body.CopyTo(combined.AsSpan(prefixBytes.Length));
        return combined;
    }

    private static int _keepAliveMax;       // 0 = unlimited; otherwise close after N requests on the same conn.
    private static int _recvTimeoutMs;      // 0 = no timeout; otherwise SO_RCVTIMEO on accepted sockets.

    public static void Run(string udsPath, IFraudScorer scorer, JsonVectorizer vectorizer)
    {
        if (File.Exists(udsPath)) File.Delete(udsPath);
        var dir = Path.GetDirectoryName(udsPath);
        if (!string.IsNullOrEmpty(dir)) Directory.CreateDirectory(dir);

        var listener = new Socket(AddressFamily.Unix, SocketType.Stream, ProtocolType.Unspecified);
        listener.Bind(new UnixDomainSocketEndPoint(udsPath));
        listener.Listen(Backlog);

        // Worker pool: N threads each block on listener.Accept(); the Linux kernel
        // load-balances incoming connections across waiters. The same thread that
        // accepts also runs the keep-alive handler to completion — no socket
        // handover, no work-queue, no scheduler ping-pong. With CPU pinned to one
        // core per replica, idle accept-blocked threads cost effectively nothing
        // until a connection arrives.
        //
        // Connection-vs-worker arithmetic: if k6 opens M concurrent VUs and N workers
        // are busy on existing keep-alive connections, the M-N excedents wait in the
        // kernel's listen backlog (no CPU/RSS in our process). KEEPALIVE_MAX bounds
        // how long a worker stays grudado in the same connection so backlog gets drained.
        int workers = ParseInt(Environment.GetEnvironmentVariable("RAW_HTTP_WORKERS"), 4);
        if (workers < 1) workers = 1;
        _keepAliveMax  = Math.Max(0, ParseInt(Environment.GetEnvironmentVariable("RAW_HTTP_KEEPALIVE_MAX"),  0));
        _recvTimeoutMs = Math.Max(0, ParseInt(Environment.GetEnvironmentVariable("RAW_HTTP_RECV_TIMEOUT_MS"), 0));

        Console.WriteLine($"RawHttpServer: listening on unix:{udsPath}, workers={workers}, keepalive_max={_keepAliveMax}, recv_timeout_ms={_recvTimeoutMs}");

        for (int i = 1; i < workers; i++)
        {
            var t = new Thread(() => WorkerLoop(listener, scorer, vectorizer))
            {
                IsBackground = true,
                Name = $"raw-http-{i}",
            };
            t.Start();
        }
        // Run worker 0 on the calling thread (Main) so the process stays alive without an extra blocker.
        WorkerLoop(listener, scorer, vectorizer);
    }

    private static void WorkerLoop(Socket listener, IFraudScorer scorer, JsonVectorizer vectorizer)
    {
        while (true)
        {
            Socket conn;
            try { conn = listener.Accept(); }
            catch (SocketException) { continue; }
            catch (ObjectDisposedException) { return; }

            if (_recvTimeoutMs > 0)
            {
                try { conn.ReceiveTimeout = _recvTimeoutMs; } catch { /* best-effort */ }
            }

            // Same thread: accept → handle → loop. The connection is kept alive across
            // many requests; only on close (or error) do we go back to Accept().
            HandleConnection(conn, scorer, vectorizer);
        }
    }

    private static int ParseInt(string? s, int fallback)
        => int.TryParse(s, out var v) ? v : fallback;

    private static void HandleConnection(Socket socket, IFraudScorer scorer, JsonVectorizer vectorizer)
    {
        int keepAliveMax = _keepAliveMax;
        try
        {
            using var s = socket;
            // Recv buffer reused across keep-alive requests on the same connection.
            Span<byte> buf = stackalloc byte[RecvBufferSize];
            int used = 0;
            int requestCount = 0;

            while (true)
            {
                // Read at least until we have headers + Content-Length bytes.
                int headerEnd, contentLength;
                while (!TryParseRequest(buf[..used], out headerEnd, out contentLength)
                       || used < headerEnd + 4 + contentLength)
                {
                    int free = buf.Length - used;
                    if (free == 0)
                    {
                        TrySend(s, BadRequestResponse);
                        return;
                    }
                    int n;
                    try { n = s.Receive(buf[used..]); }
                    catch (SocketException) { return; }
                    if (n <= 0) return;
                    used += n;
                }

                int reqLen = headerEnd + 4 + contentLength;
                HandleRequest(s, buf[..reqLen], headerEnd, contentLength, scorer, vectorizer);
                requestCount++;

                // Pipeline: shift any bytes belonging to the next request to the front.
                int leftover = used - reqLen;
                if (leftover > 0) buf[reqLen..used].CopyTo(buf);
                used = leftover;

                // Optional keep-alive cap to free this worker for the listen backlog.
                if (keepAliveMax > 0 && requestCount >= keepAliveMax) return;
            }
        }
        catch
        {
            // Swallow — connection died, nothing we can do.
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static bool TryParseRequest(ReadOnlySpan<byte> buf, out int headerEnd, out int contentLength)
    {
        headerEnd = buf.IndexOf("\r\n\r\n"u8);
        if (headerEnd < 0)
        {
            contentLength = 0;
            return false;
        }
        contentLength = ParseContentLength(buf[..headerEnd]);
        return true;
    }

    private static int ParseContentLength(ReadOnlySpan<byte> headers)
    {
        // Case-insensitive search for "content-length:" — the line we care about.
        // k6 always sends "Content-Length: <n>" so a single hardcoded prefix would
        // be enough, but a small lowercase walker is safer and still <100ns.
        for (int i = 0; i + 16 <= headers.Length; i++)
        {
            if ((headers[i] | 0x20) != (byte)'c') continue;
            if (HeaderMatches(headers, i, "content-length:"u8))
            {
                int p = i + 15;
                while (p < headers.Length && (headers[p] == (byte)' ' || headers[p] == (byte)'\t')) p++;
                int v = 0;
                while (p < headers.Length && headers[p] >= (byte)'0' && headers[p] <= (byte)'9')
                {
                    v = v * 10 + (headers[p] - (byte)'0');
                    p++;
                }
                return v;
            }
        }
        return 0;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static bool HeaderMatches(ReadOnlySpan<byte> headers, int start, ReadOnlySpan<byte> needleLowercase)
    {
        if (start + needleLowercase.Length > headers.Length) return false;
        for (int j = 0; j < needleLowercase.Length; j++)
        {
            byte c = headers[start + j];
            // Uppercase ASCII -> lowercase. Letters in the needle are already lowercase.
            if (c >= (byte)'A' && c <= (byte)'Z') c |= 0x20;
            if (c != needleLowercase[j]) return false;
        }
        return true;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void HandleRequest(
        Socket s,
        ReadOnlySpan<byte> request,
        int headerEnd,
        int contentLength,
        IFraudScorer scorer,
        JsonVectorizer vectorizer)
    {
        // Happy path: POST /fraud-score (with or without space/?/HTTP-version after).
        if (request.StartsWith("POST /fraud-score"u8))
        {
            int fraudCount;
            try
            {
                var body = request.Slice(headerEnd + 4, contentLength);
                Span<float> query = stackalloc float[Dataset.Dimensions];
                vectorizer.VectorizeJson(body, query);
                fraudCount = scorer.ScoreCount(query);
                if ((uint)fraudCount > 5u) fraudCount = fraudCount < 0 ? 0 : 5;
            }
            catch
            {
                // Malformed body / vectorizer error — return the safe default (count=0).
                TrySend(s, DefaultResponse);
                return;
            }
            TrySend(s, FraudResponses[fraudCount]);
            return;
        }

        if (request.StartsWith("GET /ready"u8))
        {
            TrySend(s, ReadyResponse);
            return;
        }

        TrySend(s, NotFoundResponse);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void TrySend(Socket s, ReadOnlySpan<byte> data)
    {
        while (data.Length > 0)
        {
            int sent;
            try { sent = s.Send(data, SocketFlags.None); }
            catch (SocketException) { return; }
            if (sent <= 0) return;
            data = data[sent..];
        }
    }
}
