using System.Globalization;
using System.Runtime.Intrinsics;
using System.Text;
using System.Text.Json;
using Rinha.Api;
using Rinha.Api.Scorers;

namespace Rinha.Bench;

/// <summary>
/// Forensic dump for IVF false-negatives: for one or more queries (by id, by
/// index, or auto-discovered as the FN of a given config), prints
///   - brute float top-N neighbors (idx, dist, label, cell)
///   - top-M cells ordered by centroid-distance (the order IVF would probe in)
///   - whether the missed fraud neighbor's cell falls within the first NP cells
///     (stage1) and/or within the first BNP cells (stage2/borderline).
///
/// Standalone — does not modify IvfScorer. Read-only diagnostic.
///
/// Usage:
///   Rinha.Bench --dump-fn --test-data=/tmp/rinha-eval/test/test-data.json \
///       [--id=tx-...] [--index=N] [--auto-config=NP=8,KP=32,BNP=32,BKP=128] \
///       [--top=64] [--cells=48] [--np=8] [--bnp=32]
/// </summary>
public static class DumpFn
{
    public static int Run(string[] args)
    {
        string testData = "/tmp/rinha-eval/test/test-data.json";
        string dataDir = Path.Combine(FindRepoRoot(), "data");
        string? id = null;
        int? index = null;
        string? autoCfg = null;
        int top = 64;
        int cells = 48;
        int np = 8;
        int bnp = 32;
        int limit = int.MaxValue;

        foreach (var a in args)
        {
            if (a.StartsWith("--test-data=")) testData = a[12..];
            else if (a.StartsWith("--data-dir=")) dataDir = a[11..];
            else if (a.StartsWith("--id=")) id = a[5..];
            else if (a.StartsWith("--index=")) index = int.Parse(a[8..]);
            else if (a.StartsWith("--auto-config=")) autoCfg = a[14..];
            else if (a.StartsWith("--top=")) top = int.Parse(a[6..]);
            else if (a.StartsWith("--cells=")) cells = int.Parse(a[8..]);
            else if (a.StartsWith("--np=")) np = int.Parse(a[5..]);
            else if (a.StartsWith("--bnp=")) bnp = int.Parse(a[6..]);
            else if (a.StartsWith("--limit=")) limit = int.Parse(a[8..]);
        }

        var vec = Path.Combine(dataDir, "references.bin");
        var lab = Path.Combine(dataDir, "labels.bin");
        var q8 = Path.Combine(dataDir, "references_q8.bin");
        var q16 = Path.Combine(dataDir, "references_q16.bin");
        var cents = Path.Combine(dataDir, "ivf_centroids.bin");
        var offs = Path.Combine(dataDir, "ivf_offsets.bin");

        if (!File.Exists(vec) || !File.Exists(cents))
        {
            Console.Error.WriteLine($"Missing data in {dataDir}");
            return 2;
        }

        using var dataset = Dataset.Open(vec, lab,
            File.Exists(q8) ? q8 : null, null,
            File.Exists(q16) ? q16 : null,
            cents, File.Exists(offs) ? offs : null);

        var root = FindRepoRoot();
        var norm = NormalizationConstants.Load(Path.Combine(root, "resources/normalization.json"));
        var mcc = MccRiskTable.Load(Path.Combine(root, "resources/mcc_risk.json"));
        var jvec = new JsonVectorizer(norm, mcc);

        Console.Error.WriteLine($"Loading {testData} ...");
        var queries = LoadQueries(testData, jvec, limit);
        Console.Error.WriteLine($"Loaded {queries.Count} queries.");

        // Build index→cell map (which IVF cell each reference vector lives in).
        var cellOf = BuildCellMap(dataset);

        // Discover which queries to dump.
        var targets = new List<int>();
        if (index.HasValue) targets.Add(index.Value);
        if (id is not null)
        {
            for (int i = 0; i < queries.Count; i++)
                if (queries[i].Id == id) { targets.Add(i); break; }
            if (targets.Count == 0)
            {
                Console.Error.WriteLine($"id '{id}' not found");
                return 1;
            }
        }
        if (autoCfg is not null)
        {
            var cfg = ParseConfig(autoCfg);
            Environment.SetEnvironmentVariable("IVF_BORDERLINE_NPROBE", cfg.bnp.ToString());
            Environment.SetEnvironmentVariable("IVF_BORDERLINE_RERANK", cfg.bkp.ToString());
            Environment.SetEnvironmentVariable("IVF_Q16", "1");
            var brute = new BruteForceScorer(dataset);
            var ivf = new IvfScorer(dataset, nProbe: cfg.np, kPrime: cfg.kp,
                earlyStop: true, earlyStopPct: 75, bboxGuided: false);
            const float thr = 0.6f;
            for (int i = 0; i < queries.Count; i++)
            {
                var sb = brute.Score(queries[i].Vec);
                var si = ivf.Score(queries[i].Vec);
                if ((sb < thr) != (si < thr)) targets.Add(i);
            }
            Console.Error.WriteLine($"auto-config FN/FP discovered: {targets.Count}");
        }

        if (targets.Count == 0)
        {
            Console.Error.WriteLine("No targets selected. Provide --id, --index, or --auto-config.");
            return 1;
        }

        foreach (var t in targets)
        {
            DumpOne(dataset, queries[t], t, top, cells, np, bnp, cellOf);
        }
        return 0;
    }

    private static unsafe void DumpOne(
        Dataset dataset, QueryRow q, int qIdx, int top, int cells, int np, int bnp, int[] cellOf)
    {
        Console.WriteLine($"=== query #{qIdx} id={q.Id} ===");
        Console.WriteLine($"  features = [{string.Join(", ", q.Vec[..Dataset.Dimensions].Select(x => x.ToString("F4", CultureInfo.InvariantCulture)))}]");

        // 1) brute top-N (sorted by float L2)
        Span<float> qf = stackalloc float[Dataset.PaddedDimensions];
        for (int i = 0; i < Dataset.Dimensions; i++) qf[i] = q.Vec[i];
        var dists = new float[dataset.Count];
        fixed (float* qPtr = qf)
        {
            var v0 = Vector256.Load(qPtr);
            var v1 = Vector256.Load(qPtr + 8);
            var vp = dataset.VectorsPtr;
            for (int i = 0; i < dataset.Count; i++)
            {
                float* row = vp + (long)i * Dataset.PaddedDimensions;
                var r0 = Vector256.Load(row);
                var r1 = Vector256.Load(row + 8);
                var d0 = r0 - v0;
                var d1 = r1 - v1;
                var s = (d0 * d0) + (d1 * d1);
                dists[i] = Vector256.Sum(s);
            }
        }
        var idxOrder = Enumerable.Range(0, dataset.Count).ToArray();
        Array.Sort(dists, idxOrder);

        Console.WriteLine($"  brute top-{Math.Min(top, dataset.Count)} (rank/idx/dist/label/cell):");
        var labels = dataset.LabelsPtr;
        int fraudsTop5 = 0;
        for (int r = 0; r < Math.Min(top, dataset.Count); r++)
        {
            int idx = idxOrder[r];
            byte lab = labels[idx];
            int cell = cellOf[idx];
            string mark = r < 5 ? "*" : " ";
            if (r < 5 && lab != 0) fraudsTop5++;
            Console.WriteLine($"    {mark}{r,3}  idx={idx,7}  d²={dists[r]:F6}  lab={lab}  cell={cell}");
        }
        Console.WriteLine($"  brute top-5 frauds = {fraudsTop5}/5  → score = {fraudsTop5 / 5f:F2}");

        // 2) cell probe order by centroid distance
        Span<float> centDist = stackalloc float[dataset.NumCells];
        fixed (float* qPtr2 = qf)
        {
            var v0 = Vector256.Load(qPtr2);
            var v1 = Vector256.Load(qPtr2 + 8);
            var cp = dataset.CentroidsPtr;
            for (int c = 0; c < dataset.NumCells; c++)
            {
                float* row = cp + (long)c * Dataset.PaddedDimensions;
                var r0 = Vector256.Load(row);
                var r1 = Vector256.Load(row + 8);
                var d0 = r0 - v0;
                var d1 = r1 - v1;
                var s = (d0 * d0) + (d1 * d1);
                centDist[c] = Vector256.Sum(s);
            }
        }
        var cellIdx = new int[dataset.NumCells];
        var centDistArr = new float[dataset.NumCells];
        for (int c = 0; c < dataset.NumCells; c++) { cellIdx[c] = c; centDistArr[c] = centDist[c]; }
        Array.Sort(centDistArr, cellIdx);

        Console.WriteLine($"  cell probe order (top-{Math.Min(cells, dataset.NumCells)} by centroid d²):");
        for (int r = 0; r < Math.Min(cells, dataset.NumCells); r++)
        {
            string stage = r < np ? "S1 " : (r < bnp ? "S2 " : "   ");
            Console.WriteLine($"    {stage}{r,3}  cell={cellIdx[r],4}  centD²={centDistArr[r]:F6}");
        }

        // 3) for the brute top-N, check whether each fraud's cell is in S1/S2/beyond
        Console.WriteLine($"  fraud-cell coverage (brute top-{top} frauds only):");
        var rank = new Dictionary<int, int>(); // cell → rank in centroid order
        for (int r = 0; r < dataset.NumCells; r++) rank[cellIdx[r]] = r;
        for (int r = 0; r < Math.Min(top, dataset.Count); r++)
        {
            int idx = idxOrder[r];
            if (labels[idx] == 0) continue;
            int cell = cellOf[idx];
            int cr = rank[cell];
            string stage = cr < np ? "S1" : (cr < bnp ? "S2" : "OUT");
            Console.WriteLine($"    rank={r,3} idx={idx,7} cell={cell,4} cellRank={cr,3} stage={stage} d²={dists[r]:F6}");
        }
        Console.WriteLine();
    }

    private static unsafe int[] BuildCellMap(Dataset dataset)
    {
        var map = new int[dataset.Count];
        var offs = dataset.CellOffsetsPtr;
        for (int c = 0; c < dataset.NumCells; c++)
        {
            int start = offs[c];
            int end = offs[c + 1];
            for (int i = start; i < end; i++) map[i] = c;
        }
        return map;
    }

    private record QueryRow(string Id, float[] Vec);

    private static List<QueryRow> LoadQueries(string path, JsonVectorizer jvec, int limit)
    {
        var bytes = File.ReadAllBytes(path);
        using var doc = JsonDocument.Parse(bytes);
        var entries = doc.RootElement.GetProperty("entries");
        var list = new List<QueryRow>(Math.Min(entries.GetArrayLength(), limit));
        int n = 0;
        foreach (var entry in entries.EnumerateArray())
        {
            if (n >= limit) break;
            var req = entry.GetProperty("request");
            var raw = Encoding.UTF8.GetBytes(req.GetRawText());
            var vec = new float[Dataset.Dimensions];
            jvec.VectorizeJson(raw, vec);
            string id = req.TryGetProperty("id", out var idEl) ? idEl.GetString() ?? "" : "";
            list.Add(new QueryRow(id, vec));
            n++;
        }
        return list;
    }

    private static (int np, int kp, int bnp, int bkp) ParseConfig(string s)
    {
        int np = 8, kp = 32, bnp = 32, bkp = 0;
        foreach (var part in s.Split(','))
        {
            var kv = part.Split('=');
            if (kv.Length != 2) continue;
            int v = int.Parse(kv[1], CultureInfo.InvariantCulture);
            switch (kv[0].Trim().ToUpperInvariant())
            {
                case "NP": case "NPROBE": np = v; break;
                case "KP": case "KPRIME": kp = v; break;
                case "BNP": case "BORDERLINE_NPROBE": bnp = v; break;
                case "BKP": case "BORDERLINE_RERANK": case "BORDERLINE_KPRIME": bkp = v; break;
            }
        }
        return (np, kp, bnp, bkp);
    }

    private static string FindRepoRoot()
    {
        var dir = AppContext.BaseDirectory;
        while (dir is not null && !File.Exists(Path.Combine(dir, "Rinha.slnx")))
            dir = Path.GetDirectoryName(dir);
        return dir ?? throw new InvalidOperationException("Could not locate repo root");
    }
}
