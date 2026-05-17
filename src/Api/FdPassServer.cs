using System.Net.Sockets;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Microsoft.Win32.SafeHandles;

namespace Rinha.Api;

/// <summary>
/// Receives client TCP file descriptors from the LB via SCM_RIGHTS over a
/// single persistent UDS control connection. Each extracted fd is wrapped
/// into a <see cref="Socket"/> and handed to the existing
/// <see cref="RawHttpServer.HandleConnectionPublic"/> keep-alive loop on the
/// ThreadPool.
///
/// Layout of msghdr / cmsghdr matches Linux x86_64 kernel/glibc (56 / 16 bytes).
/// Verified compatible with jrblatt/so-no-forevis:v1.0.0 which switched from
/// HTTP byte-proxy to fd-passing.
/// </summary>
internal static class FdPassServer
{
    private const int SolSocket = 1;
    private const int ScmRights = 1;
    private const int IpprotoTcp = 6;
    private const int TcpNodelay = 1;
    private const int ListenBacklog = 8;

    [StructLayout(LayoutKind.Sequential)]
    private unsafe struct MsgHdr
    {
        public void* msg_name;
        public uint msg_namelen;
        private uint _pad1;
        public IoVec* msg_iov;
        public nuint msg_iovlen;
        public void* msg_control;
        public nuint msg_controllen;
        public int msg_flags;
        private int _pad2;
    }

    [StructLayout(LayoutKind.Sequential)]
    private unsafe struct IoVec { public void* iov_base; public nuint iov_len; }

    [StructLayout(LayoutKind.Sequential)]
    private struct CmsgHdr { public nuint cmsg_len; public int cmsg_level; public int cmsg_type; }

    [DllImport("libc", SetLastError = true)]
    private static extern unsafe nint recvmsg(int sockfd, MsgHdr* msg, int flags);

    [DllImport("libc", SetLastError = true)]
    private static extern unsafe int setsockopt(int sockfd, int level, int optname, int* optval, uint optlen);

    public static void Run(string udsPath, IFraudScorer scorer, JsonVectorizer vectorizer, SelectiveDecisionCascade selectiveCascade)
    {
        // The LB connects to "<udsPath>.ctrl" — convention from
        // jrblatt/so-no-forevis (uds::connect_ctrl(format!("{path}.ctrl"))).
        var ctrlPath = udsPath + ".ctrl";
        if (File.Exists(ctrlPath)) File.Delete(ctrlPath);
        var dir = Path.GetDirectoryName(ctrlPath);
        if (!string.IsNullOrEmpty(dir)) Directory.CreateDirectory(dir);

        var listener = new Socket(AddressFamily.Unix, SocketType.Stream, ProtocolType.Unspecified);
        listener.Bind(new UnixDomainSocketEndPoint(ctrlPath));
        listener.Listen(ListenBacklog);

        try
        {
            File.SetUnixFileMode(ctrlPath,
                UnixFileMode.UserRead  | UnixFileMode.UserWrite  |
                UnixFileMode.GroupRead | UnixFileMode.GroupWrite |
                UnixFileMode.OtherRead | UnixFileMode.OtherWrite);
        }
        catch (Exception ex) { Console.Error.WriteLine($"chmod ctrl uds failed: {ex.Message}"); }

        Console.WriteLine($"FdPassServer: listening on unix:{ctrlPath}, awaiting LB control connections");

        // Each LB worker opens its own control connection. We accept them all and
        // run a RecvLoop per control fd on a dedicated thread (the loop just
        // recvmsg's and queues fds to the ThreadPool — single-threaded per ctrl
        // is fine).
        while (true)
        {
            Socket ctrl;
            try { ctrl = listener.Accept(); }
            catch (SocketException) { continue; }
            catch (ObjectDisposedException) { return; }

            Console.WriteLine("FdPassServer: LB control connection established");

            var ctrlState = new CtrlState(ctrl, scorer, vectorizer, selectiveCascade);
            var t = new Thread(static state =>
            {
                var st = (CtrlState)state!;
                int ctrlFd = (int)st.Ctrl.SafeHandle.DangerousGetHandle();
                try { RecvLoop(ctrlFd, st.Scorer, st.Vectorizer, st.SelectiveCascade); }
                finally { try { st.Ctrl.Dispose(); } catch { } }
                Console.Error.WriteLine("FdPassServer: control channel closed");
            })
            {
                IsBackground = true,
                Name = "fdpass-ctrl",
            };
            t.Start(ctrlState);
        }
    }

    private sealed class CtrlState
    {
        public readonly Socket Ctrl;
        public readonly IFraudScorer Scorer;
        public readonly JsonVectorizer Vectorizer;
        public readonly SelectiveDecisionCascade SelectiveCascade;
        public CtrlState(Socket c, IFraudScorer s, JsonVectorizer v, SelectiveDecisionCascade sc)
        { Ctrl = c; Scorer = s; Vectorizer = v; SelectiveCascade = sc; }
    }

    private static unsafe void RecvLoop(int ctrlFd, IFraudScorer scorer, JsonVectorizer vectorizer, SelectiveDecisionCascade selectiveCascade)
    {
        const int CmsgBufSize = 24; // CMSG_SPACE(sizeof(int)) on x86_64
        byte dummy = 0;
        byte* cmsgBuf = stackalloc byte[CmsgBufSize];

        while (true)
        {
            var iov = new IoVec { iov_base = &dummy, iov_len = 1 };
            new Span<byte>(cmsgBuf, CmsgBufSize).Clear();

            var msg = new MsgHdr
            {
                msg_iov = &iov,
                msg_iovlen = 1,
                msg_control = cmsgBuf,
                msg_controllen = CmsgBufSize,
            };

            nint r = recvmsg(ctrlFd, &msg, 0);
            if (r <= 0) return; // EOF or error -> control connection gone

            var hdr = (CmsgHdr*)cmsgBuf;
            if (hdr->cmsg_level != SolSocket || hdr->cmsg_type != ScmRights) continue;

            int clientFd = *(int*)(cmsgBuf + sizeof(CmsgHdr));
            if (clientFd < 0) continue;

            // TCP_NODELAY on the client TCP fd (one-shot per connection).
            int one = 1;
            setsockopt(clientFd, IpprotoTcp, TcpNodelay, &one, 4);

            // Wrap raw fd in a Socket. ownsHandle:true -> Socket.Dispose() closes it.
            Socket sock;
            try
            {
                var safe = new SafeSocketHandle((IntPtr)clientFd, ownsHandle: true);
                sock = new Socket(safe);
            }
            catch
            {
                continue;
            }

            var state = new HandoffState(sock, scorer, vectorizer, selectiveCascade);
            ThreadPool.UnsafeQueueUserWorkItem(s_handoff, state, preferLocal: false);
        }
    }

    private sealed class HandoffState
    {
        public readonly Socket Sock;
        public readonly IFraudScorer Scorer;
        public readonly JsonVectorizer Vectorizer;
        public readonly SelectiveDecisionCascade SelectiveCascade;
        public HandoffState(Socket s, IFraudScorer sc, JsonVectorizer v, SelectiveDecisionCascade c)
        { Sock = s; Scorer = sc; Vectorizer = v; SelectiveCascade = c; }
    }

    private static readonly Action<HandoffState> s_handoff = static st =>
        RawHttpServer.HandleConnectionPublic(st.Sock, st.Scorer, st.Vectorizer, st.SelectiveCascade);
}
