using System.Diagnostics;
using System.Text;
using System.Text.Json;
using Rinha.Api;

namespace Rinha.Bench;

/// <summary>Micro-bench: time JsonVectorizer.VectorizeJson on every test-data entry,
/// reported per call. Also isolates per-field cost via targeted FastNumberParse loops.</summary>
internal static class ParseBench
{
    public static int Run(string[] args)
    {
        string testData = "bench/k6/test-data.json";
        int iters = 20;
        bool perField = false;
        foreach (var a in args)
        {
            if (a.StartsWith("--test-data=")) testData = a["--test-data=".Length..];
            else if (a.StartsWith("--iters=")) iters = int.Parse(a["--iters=".Length..]);
            else if (a == "--per-field") perField = true;
        }

        var bytes = File.ReadAllBytes(testData);
        using var doc = JsonDocument.Parse(bytes);
        var entries = doc.RootElement.GetProperty("entries");
        var bodies = new List<byte[]>(entries.GetArrayLength());
        foreach (var e in entries.EnumerateArray())
            bodies.Add(Encoding.UTF8.GetBytes(e.GetProperty("request").GetRawText()));

        var norm = NormalizationConstants.Load("resources/normalization.json");
        var mcc = MccRiskTable.Load("resources/mcc_risk.json");
        var jvec = new JsonVectorizer(norm, mcc);

        var floatBuf = new float[Dataset.Dimensions];
        for (int w = 0; w < 3; w++)
            foreach (var b in bodies) jvec.VectorizeJson(b, floatBuf);

        var times = new long[iters];
        var sw = new Stopwatch();
        long total = 0;
        for (int it = 0; it < iters; it++)
        {
            sw.Restart();
            foreach (var b in bodies) jvec.VectorizeJson(b, floatBuf);
            sw.Stop();
            times[it] = sw.ElapsedTicks;
            total += sw.ElapsedTicks;
        }

        double tickToNs = 1_000_000_000.0 / Stopwatch.Frequency;
        Array.Sort(times);
        double avgNs = (total * tickToNs) / iters / bodies.Count;
        double minNs = (times[0] * tickToNs) / bodies.Count;
        double medNs = (times[iters / 2] * tickToNs) / bodies.Count;
        Console.WriteLine($"JSON_FAST={Environment.GetEnvironmentVariable("JSON_FAST") ?? "1(default)"}  N={bodies.Count}  iters={iters}");
        Console.WriteLine($"per-parse: avg={avgNs:F1}ns  min={minNs:F1}ns  median={medNs:F1}ns");
        Console.WriteLine($"total/iter: min={times[0]*tickToNs/1e6:F2}ms  med={times[iters/2]*tickToNs/1e6:F2}ms");

        if (perField)
            RunPerField(bodies, tickToNs, iters);
        return 0;
    }

    // Per-field isolation: pre-extract each field's raw byte slice, then time
    // parsing each list separately. This removes the cross-field cost (Compose,
    // ParseIsoUtc, ScanKnownMerchants, etc.) so we see only the number scan.
    private static void RunPerField(List<byte[]> bodies, double tickToNs, int iters)
    {
        Console.WriteLine("\n--- Per-field isolation ---");
        var f = ExtractFields(bodies);
        MeasureField("amount       (double)", f.amount, false, tickToNs, iters);
        MeasureField("installments (int)   ", f.installments, true, tickToNs, iters);
        MeasureField("custAvg      (double)", f.custAvg, false, tickToNs, iters);
        MeasureField("txCount24h   (int)   ", f.txCount, true, tickToNs, iters);
        MeasureField("merchAvg     (double)", f.merchAvg, false, tickToNs, iters);
        MeasureField("kmFromHome   (double)", f.kmHome, false, tickToNs, iters);
        MeasureField("kmFromCurrent(double)", f.kmCurrent, false, tickToNs, iters);
        MeasureIso  ("requested_at (iso)   ", f.iso, tickToNs, iters);
        MeasureMcc  ("mcc          (q16)   ", f.mcc, tickToNs, iters);
        MeasureKM   ("knownMerchants scan  ", f.kmPairs, tickToNs, iters);
    }

    private static void MeasureIso(string label, List<byte[]> samples, double tickToNs, int iters)
    {
        // Replicate ParseIsoUtc's hot path body.
        for (int w = 0; w < 3; w++) DrainIso(samples);
        var times = new long[iters];
        var sw = new Stopwatch();
        for (int it = 0; it < iters; it++) { sw.Restart(); DrainIso(samples); sw.Stop(); times[it] = sw.ElapsedTicks; }
        Array.Sort(times);
        double minNs = (times[0] * tickToNs) / samples.Count;
        double medNs = (times[iters / 2] * tickToNs) / samples.Count;
        Console.WriteLine($"  {label}: min={minNs:F1}ns med={medNs:F1}ns  (total med={times[iters/2]*tickToNs/1e6:F2}ms over {samples.Count} samples)");
    }
    private static void DrainIso(List<byte[]> samples)
    {
        long sink = 0;
        foreach (var s in samples)
        {
            int y = (s[0]-'0')*1000 + (s[1]-'0')*100 + (s[2]-'0')*10 + (s[3]-'0');
            int mo = (s[5]-'0')*10 + (s[6]-'0');
            int d = (s[8]-'0')*10 + (s[9]-'0');
            int h = (s[11]-'0')*10 + (s[12]-'0');
            int mi = (s[14]-'0')*10 + (s[15]-'0');
            int se = (s[17]-'0')*10 + (s[18]-'0');
            var dt = new DateTime(y, mo, d, h, mi, se, DateTimeKind.Utc);
            sink += dt.Ticks + (int)dt.DayOfWeek;
        }
        s_sink ^= sink;
    }

    private static void MeasureMcc(string label, List<byte[]> samples, double tickToNs, int iters)
    {
        var norm = NormalizationConstants.Load("resources/normalization.json");
        var mcc = MccRiskTable.Load("resources/mcc_risk.json");
        for (int w = 0; w < 3; w++) DrainMcc(mcc, samples);
        var times = new long[iters];
        var sw = new Stopwatch();
        for (int it = 0; it < iters; it++) { sw.Restart(); DrainMcc(mcc, samples); sw.Stop(); times[it] = sw.ElapsedTicks; }
        Array.Sort(times);
        double minNs = (times[0] * tickToNs) / samples.Count;
        double medNs = (times[iters / 2] * tickToNs) / samples.Count;
        Console.WriteLine($"  {label}: min={minNs:F1}ns med={medNs:F1}ns  (total med={times[iters/2]*tickToNs/1e6:F2}ms over {samples.Count} samples)");
    }
    private static void DrainMcc(MccRiskTable t, List<byte[]> samples)
    {
        long sink = 0;
        foreach (var s in samples) sink += t.GetQ16(s);
        s_sink ^= sink;
    }

    private static void MeasureKM(string label, List<(byte[] arr, byte[] id)> samples, double tickToNs, int iters)
    {
        for (int w = 0; w < 3; w++) DrainKM(samples);
        var times = new long[iters];
        var sw = new Stopwatch();
        for (int it = 0; it < iters; it++) { sw.Restart(); DrainKM(samples); sw.Stop(); times[it] = sw.ElapsedTicks; }
        Array.Sort(times);
        double minNs = (times[0] * tickToNs) / samples.Count;
        double medNs = (times[iters / 2] * tickToNs) / samples.Count;
        Console.WriteLine($"  {label}: min={minNs:F1}ns med={medNs:F1}ns  (total med={times[iters/2]*tickToNs/1e6:F2}ms over {samples.Count} samples)");
    }
    private static void DrainKM(List<(byte[] arr, byte[] id)> samples)
    {
        long sink = 0;
        foreach (var (arr, id) in samples)
        {
            ReadOnlySpan<byte> a = arr; ReadOnlySpan<byte> needle = id;
            int i = 0;
            while (i < a.Length)
            {
                while (i < a.Length && a[i] != (byte)'"') i++;
                if (i >= a.Length) break;
                int start = ++i;
                while (i < a.Length && a[i] != (byte)'"') i++;
                if (i >= a.Length) break;
                if (a.Slice(start, i - start).SequenceEqual(needle)) { sink++; break; }
                i++;
            }
        }
        s_sink ^= sink;
    }

    private static void MeasureField(string label, List<byte[]> samples, bool isInt, double tickToNs, int iters)
    {
        var pads = samples.Select(s => { var p = new byte[s.Length + 16]; s.CopyTo(p, 0); p[s.Length] = (byte)','; return p; }).ToArray();
        for (int w = 0; w < 3; w++) DrainOne(pads, isInt);
        var times = new long[iters];
        var sw = new Stopwatch();
        for (int it = 0; it < iters; it++)
        {
            sw.Restart();
            DrainOne(pads, isInt);
            sw.Stop();
            times[it] = sw.ElapsedTicks;
        }
        Array.Sort(times);
        double minNs = (times[0] * tickToNs) / samples.Count;
        double medNs = (times[iters / 2] * tickToNs) / samples.Count;
        Console.WriteLine($"  {label}: min={minNs:F1}ns med={medNs:F1}ns  (total med={times[iters/2]*tickToNs/1e6:F2}ms over {samples.Count} samples)");
    }

    private static long s_sink;
    private static void DrainOne(byte[][] pads, bool isInt)
    {
        long sink = 0;
        if (isInt)
        {
            foreach (var p in pads) { int pos = 0; sink += FastNumberParse.ParseInt32(p, ref pos); }
        }
        else
        {
            foreach (var p in pads) { int pos = 0; sink += (long)FastNumberParse.ParseDouble(p, ref pos); }
        }
        s_sink ^= sink;
    }

    private static (List<byte[]> amount, List<byte[]> installments, List<byte[]> custAvg, List<byte[]> txCount, List<byte[]> merchAvg, List<byte[]> kmHome, List<byte[]> kmCurrent, List<byte[]> iso, List<byte[]> mcc, List<(byte[] arr, byte[] id)> kmPairs)
        ExtractFields(List<byte[]> bodies)
    {
        var amount = new List<byte[]>(bodies.Count);
        var installments = new List<byte[]>(bodies.Count);
        var custAvg = new List<byte[]>(bodies.Count);
        var txCount = new List<byte[]>(bodies.Count);
        var merchAvg = new List<byte[]>(bodies.Count);
        var kmHome = new List<byte[]>(bodies.Count);
        var kmCurrent = new List<byte[]>(bodies.Count);
        var iso = new List<byte[]>(bodies.Count);
        var mcc = new List<byte[]>(bodies.Count);
        var kmPairs = new List<(byte[], byte[])>(bodies.Count);

        foreach (var b in bodies)
        {
            string s = Encoding.UTF8.GetString(b);
            amount.Add(Slice(s, "\"amount\":", ','));
            installments.Add(Slice(s, "\"installments\":", ','));
            int ci = s.IndexOf("\"customer\":");
            int mi = s.IndexOf("\"merchant\":");
            custAvg.Add(Slice(s, "\"avg_amount\":", ',', ci));
            txCount.Add(Slice(s, "\"tx_count_24h\":", ','));
            merchAvg.Add(Slice(s, "\"avg_amount\":", '}', mi));
            kmHome.Add(Slice(s, "\"km_from_home\":", '}'));
            int li = s.IndexOf("\"last_transaction\":{");
            if (li >= 0) kmCurrent.Add(Slice(s, "\"km_from_current\":", '}', li));
            iso.Add(Slice(s, "\"requested_at\":\"", '"'));
            mcc.Add(Slice(s, "\"mcc\":\"", '"'));
            // known_merchants array + merchant.id
            int kstart = s.IndexOf("\"known_merchants\":[") + "\"known_merchants\":".Length;
            int kend = s.IndexOf(']', kstart) + 1;
            int midi = s.IndexOf("\"merchant\":{\"id\":\"") + "\"merchant\":{\"id\":\"".Length;
            int midend = s.IndexOf('"', midi);
            kmPairs.Add((Encoding.UTF8.GetBytes(s[kstart..kend]), Encoding.UTF8.GetBytes(s[midi..midend])));
        }
        return (amount, installments, custAvg, txCount, merchAvg, kmHome, kmCurrent, iso, mcc, kmPairs);
    }

    private static byte[] Slice(string s, string key, char delim, int from = 0)
    {
        int i = s.IndexOf(key, from) + key.Length;
        int j = s.IndexOf(delim, i);
        return Encoding.UTF8.GetBytes(s[i..j]);
    }
}
