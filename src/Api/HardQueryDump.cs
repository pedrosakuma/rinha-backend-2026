using System.Globalization;
using System.Text;

namespace Rinha.Api;

/// <summary>
/// J24: offline dataset collector for the hard-query predictor.
/// Enabled when env HARDQ_DUMP_PATH is set. Each call to Append() writes one CSV
/// row "f0,f1,...,f13,rowsScanned,earlyStopMode" to the configured file.
///
/// A shared StringBuilder under a single lock is used — slow but guarantees no
/// data loss when the container is killed. Use only for data-collection runs.
/// </summary>
public static class HardQueryDump
{
    private static FileStream? s_file;
    private static readonly object s_lock = new();
    private static readonly StringBuilder s_buf = new(64 * 1024);
    private const int FlushBytes = 32 * 1024;

    public static bool Enabled => s_file is not null;

    public static void Open(string path)
    {
        // Replicas share the volume; suffix with hostname to avoid clobbering.
        var host = Environment.GetEnvironmentVariable("HOSTNAME") ?? "default";
        var dir = Path.GetDirectoryName(path) ?? ".";
        var name = Path.GetFileNameWithoutExtension(path);
        var ext = Path.GetExtension(path);
        var resolved = Path.Combine(dir, $"{name}-{host}{ext}");
        s_file = new FileStream(resolved, FileMode.Create, FileAccess.Write, FileShare.Read);
    }

    public static void Append(ReadOnlySpan<float> features, int rowsScanned, int earlyStopMode)
    {
        if (s_file is null) return;
        var inv = CultureInfo.InvariantCulture;
        lock (s_lock)
        {
            for (int i = 0; i < features.Length; i++)
            {
                s_buf.Append(features[i].ToString("0.######", inv));
                s_buf.Append(',');
            }
            s_buf.Append(rowsScanned).Append(',').Append(earlyStopMode).Append('\n');
            if (s_buf.Length >= FlushBytes) FlushLocked();
        }
    }

    private static void FlushLocked()
    {
        var bytes = Encoding.ASCII.GetBytes(s_buf.ToString());
        s_buf.Clear();
        s_file?.Write(bytes, 0, bytes.Length);
    }

    public static void Close()
    {
        lock (s_lock)
        {
            if (s_file is null) return;
            FlushLocked();
            s_file.Flush();
            s_file.Dispose();
            s_file = null;
        }
    }
}
