using System.Net.Sockets;
using System.Runtime.InteropServices;
using Microsoft.Win32.SafeHandles;

namespace Rinha.Api;

/// <summary>
/// Listens on a Unix Domain Socket at <c>{udsPath}.ctrl</c> for SCM_RIGHTS
/// file-descriptor passing from a load balancer like <c>so-no-forevis:v1.0.0</c>.
///
/// Protocol (confirmed by stracing jrblatt/so-no-forevis:v1.0.0):
///   * LB connects once per worker to <c>{sock}.ctrl</c>
///   * For each accepted client TCP connection, LB sends a single byte of
///     dummy data (0x00) plus an ancillary <c>SOL_SOCKET / SCM_RIGHTS</c>
///     control message containing exactly one int (the TCP socket FD).
///   * The API takes ownership of that FD, wraps it in a <see cref="Socket"/>,
///     and serves HTTP/1.1 directly on the TCP connection — eliminating the
///     LB-as-data-proxy hop entirely (zero copies through the LB process).
///
/// This file owns ONLY the recv-loop. Once an FD is materialized into a
/// <see cref="Socket"/>, it's handed back to the same per-connection
/// handler the regular UDS path uses (see <see cref="RawHttpServer"/>).
/// </summary>
internal static class FdPassingListener
{
    private const int Backlog = 64;
    private const int MaxFdsPerMsg = 8;

    public static void Run(string ctrlPath, Action<Socket> onSocketReceived)
    {
        if (File.Exists(ctrlPath)) File.Delete(ctrlPath);
        var dir = Path.GetDirectoryName(ctrlPath);
        if (!string.IsNullOrEmpty(dir)) Directory.CreateDirectory(dir);

        var listener = new Socket(AddressFamily.Unix, SocketType.Stream, ProtocolType.Unspecified);
        listener.Bind(new UnixDomainSocketEndPoint(ctrlPath));
        listener.Listen(Backlog);

        // chmod 0666 so the LB container (different uid) can connect.
        try
        {
            File.SetUnixFileMode(ctrlPath,
                UnixFileMode.UserRead  | UnixFileMode.UserWrite  |
                UnixFileMode.GroupRead | UnixFileMode.GroupWrite |
                UnixFileMode.OtherRead | UnixFileMode.OtherWrite);
        }
        catch { /* best-effort */ }

        Console.WriteLine($"FdPassingListener: listening on unix:{ctrlPath} (SCM_RIGHTS)");

        // One accept thread; each control connection gets its own recv-loop thread.
        // In practice the LB opens exactly one ctrl conn per worker.
        while (true)
        {
            Socket ctrl;
            try { ctrl = listener.Accept(); }
            catch (SocketException) { continue; }
            catch (ObjectDisposedException) { return; }

            var t = new Thread(static obj =>
            {
                var (sock, cb) = ((Socket, Action<Socket>))obj!;
                try { RecvLoop(sock, cb); }
                catch (Exception ex)
                {
                    Console.Error.WriteLine($"FdPassingListener recv loop ended: {ex.Message}");
                }
            })
            { IsBackground = true, Name = "raw-http-fd-recv" };
            t.Start((ctrl, onSocketReceived));
        }
    }

    private static void RecvLoop(Socket ctrl, Action<Socket> onSocketReceived)
    {
        using var _ = ctrl;
        int fd = (int)ctrl.Handle;

        // Buffer for the dummy data byte the LB sends along with each FD.
        Span<byte> dataBuf = stackalloc byte[16];

        // Ancillary buffer big enough for several FDs per message (the LB may
        // batch). CMSG_SPACE(N*sizeof(int)) on glibc x86_64 = ALIGN(16 + N*4).
        int cmsgSpace = LibcInterop.CmsgSpace(MaxFdsPerMsg * sizeof(int));
        Span<byte> cmsgBuf = stackalloc byte[cmsgSpace];

        Span<int> fds = stackalloc int[MaxFdsPerMsg];

        while (true)
        {
            int received = LibcInterop.RecvMsgWithFds(fd, dataBuf, cmsgBuf, fds, out int fdCount);
            if (received < 0)
            {
                int err = Marshal.GetLastPInvokeError();
                if (err == LibcInterop.EINTR) continue;
                throw new IOException($"recvmsg failed errno={err}");
            }
            if (received == 0 && fdCount == 0) return; // peer closed

            for (int i = 0; i < fdCount; i++)
            {
                int rxFd = fds[i];
                if (rxFd < 0) continue;

                Socket clientSock;
                try
                {
                    var safe = new SafeSocketHandle(new IntPtr(rxFd), ownsHandle: true);
                    clientSock = new Socket(safe);
                }
                catch
                {
                    LibcInterop.Close(rxFd);
                    continue;
                }

                onSocketReceived(clientSock);
            }
        }
    }
}

/// <summary>
/// Minimal libc P/Invoke for <c>recvmsg(2)</c> + <c>SCM_RIGHTS</c>. .NET has no
/// managed equivalent and these structs/macros aren't ABI-stable across libcs,
/// so we hand-roll the layout for Linux glibc x86_64 (matches musl too — both
/// use the kernel ABI definitions for <c>struct msghdr</c>/<c>cmsghdr</c>).
/// </summary>
internal static class LibcInterop
{
    public const int EINTR = 4;

    [StructLayout(LayoutKind.Sequential)]
    private struct IoVec
    {
        public IntPtr Base;   // void*
        public nuint  Len;    // size_t
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct MsgHdr
    {
        public IntPtr  MsgName;       // void*           (NULL — connected socket)
        public uint    MsgNameLen;    // socklen_t (4)
        // 4 bytes padding to 8-byte align iov on x86_64
        private uint   _pad0;
        public IntPtr  MsgIov;        // struct iovec*
        public nuint   MsgIovLen;     // size_t
        public IntPtr  MsgControl;    // void*
        public nuint   MsgControlLen; // size_t
        public int     MsgFlags;      // int
        // 4 bytes trailing padding (struct size 56) — fine, recvmsg only reads to MsgFlags.
    }

    // glibc CMSG_ALIGN: round to sizeof(size_t) = 8 on x86_64.
    private static int CmsgAlign(int len) => (len + (sizeof(long) - 1)) & ~(sizeof(long) - 1);
    // sizeof(struct cmsghdr) = 16 on x86_64 (size_t cmsg_len; int cmsg_level; int cmsg_type;)
    private const int CmsgHdrSize = 16;

    public static int CmsgSpace(int dataLen) => CmsgAlign(CmsgHdrSize) + CmsgAlign(dataLen);
    public static int CmsgLen(int dataLen)   => CmsgAlign(CmsgHdrSize) + dataLen;

    private const int SOL_SOCKET = 1;
    private const int SCM_RIGHTS = 1;

    [DllImport("libc", SetLastError = true, EntryPoint = "recvmsg")]
    private static extern nint recvmsg(int sockfd, ref MsgHdr msg, int flags);

    [DllImport("libc", SetLastError = true, EntryPoint = "close")]
    private static extern int close_native(int fd);

    public static int Close(int fd) => close_native(fd);

    /// <summary>
    /// Blocking recvmsg that copies any received data into <paramref name="dataBuf"/>
    /// and parses SCM_RIGHTS ancillary messages into <paramref name="fdsOut"/>.
    /// </summary>
    public static unsafe int RecvMsgWithFds(
        int sockfd,
        Span<byte> dataBuf,
        Span<byte> cmsgBuf,
        Span<int>  fdsOut,
        out int fdCount)
    {
        fdCount = 0;
        fixed (byte* dataPtr = dataBuf)
        fixed (byte* cmsgPtr = cmsgBuf)
        {
            IoVec iov;
            iov.Base = (IntPtr)dataPtr;
            iov.Len  = (nuint)dataBuf.Length;

            MsgHdr msg = default;
            msg.MsgIov        = (IntPtr)(&iov);
            msg.MsgIovLen     = 1;
            msg.MsgControl    = (IntPtr)cmsgPtr;
            msg.MsgControlLen = (nuint)cmsgBuf.Length;

            nint n = recvmsg(sockfd, ref msg, 0);
            if (n < 0) return -1;

            // Walk cmsg list manually (CMSG_FIRSTHDR / CMSG_NXTHDR equivalent).
            int controlLen = (int)msg.MsgControlLen;
            int offset = 0;
            while (offset + CmsgHdrSize <= controlLen)
            {
                // struct cmsghdr layout on x86_64:
                //   size_t cmsg_len;   // 8 bytes
                //   int    cmsg_level; // 4 bytes
                //   int    cmsg_type;  // 4 bytes
                long cmsgLen   = *(long*)(cmsgPtr + offset);
                int  cmsgLevel = *(int*) (cmsgPtr + offset + 8);
                int  cmsgType  = *(int*) (cmsgPtr + offset + 12);

                if (cmsgLen < CmsgHdrSize) break;
                if (offset + (int)cmsgLen > controlLen) break;

                if (cmsgLevel == SOL_SOCKET && cmsgType == SCM_RIGHTS)
                {
                    int dataLen = (int)cmsgLen - CmsgHdrSize;
                    int nFds = dataLen / sizeof(int);
                    int* fdData = (int*)(cmsgPtr + offset + CmsgHdrSize);
                    for (int i = 0; i < nFds && fdCount < fdsOut.Length; i++)
                        fdsOut[fdCount++] = fdData[i];
                }

                offset += CmsgAlign((int)cmsgLen);
            }

            return (int)n;
        }
    }
}
