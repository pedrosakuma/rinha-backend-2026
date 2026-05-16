using System.Diagnostics;
using System.Text;
using System.Text.Json;
using Rinha.Api;

namespace Rinha.Bench;

/// <summary>Micro-bench: time JsonVectorizer.VectorizeJson on every test-data entry,
/// reported per call. Toggles JSON_FAST internally for A/B.</summary>
internal static class ParseBench
{
    public static int Run(string[] args)
    {
        string testData = "bench/k6/test-data.json";
        int iters = 20;
        foreach (var a in args)
        {
            if (a.StartsWith("--test-data=")) testData = a["--test-data=".Length..];
            else if (a.StartsWith("--iters=")) iters = int.Parse(a["--iters=".Length..]);
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

        // Warmup (forces JIT, primes caches).
        var floatBuf = new float[Dataset.Dimensions];
        for (int w = 0; w < 3; w++)
            foreach (var b in bodies) jvec.VectorizeJson(b, floatBuf);

        // Measure.
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
        return 0;
    }
}
