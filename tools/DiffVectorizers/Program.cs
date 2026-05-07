using System.Text.Json;
using Rinha.Api;

namespace Rinha.Api
{
    internal static class Dataset
    {
        public const int Dimensions = 14;
        public const int PaddedDimensions = 16;
    }
}

namespace Rinha.Tools.DiffVectorizers
{
    internal static class Program
{
    static int Main(string[] args)
    {
        if (args.Length < 3)
        {
            Console.Error.WriteLine("usage: DiffVectorizers <normalization.json> <mcc_risk.json> <test-data.json>");
            return 2;
        }
        var norm = NormalizationConstants.Load(args[0]);
        var mcc = MccRiskTable.Load(args[1]);
        var v1 = new Vectorizer(norm, mcc);
        var v2 = new JsonVectorizer(norm, mcc);

        using var fs = File.OpenRead(args[2]);
        using var doc = JsonDocument.Parse(fs);
        var entries = doc.RootElement.GetProperty("entries");

        int total = 0, divergent = 0;
        Span<float> a = stackalloc float[Dataset.Dimensions];
        Span<float> b = stackalloc float[Dataset.Dimensions];
        var sample = new List<(int idx, string id, int dim, float a, float b)>();
        var perDimMaxAbs = new float[Dataset.Dimensions];

        foreach (var entry in entries.EnumerateArray())
        {
            var requestEl = entry.GetProperty("request");
            // Use GetRawText() to preserve the EXACT byte format from test-data.json,
            // matching what k6 sends (JSON.stringify of entry.request).
            var bodyBytes = System.Text.Encoding.UTF8.GetBytes(requestEl.GetRawText());
            var request = JsonSerializer.Deserialize(bodyBytes, AppJsonContext.Default.FraudRequest)!;

            v1.Vectorize(request, a);
            v2.VectorizeJson(bodyBytes, b);

            bool diff = false;
            for (int i = 0; i < Dataset.Dimensions; i++)
            {
                float d = MathF.Abs(a[i] - b[i]);
                if (d > 0f)
                {
                    diff = true;
                    if (d > perDimMaxAbs[i]) perDimMaxAbs[i] = d;
                    if (sample.Count < 30)
                        sample.Add((total, request.Id, i, a[i], b[i]));
                }
            }
            if (diff) divergent++;
            total++;
        }

        Console.WriteLine($"Total entries: {total}");
        Console.WriteLine($"Divergent rows (any dim): {divergent}");
        Console.WriteLine("Per-dim max |Δ|:");
        for (int i = 0; i < Dataset.Dimensions; i++)
            if (perDimMaxAbs[i] > 0)
                Console.WriteLine($"  dim[{i,2}] max|Δ|={perDimMaxAbs[i]:G9}");
        Console.WriteLine("Sample divergences (first 30):");
        foreach (var (idx, id, dim, av, bv) in sample)
            Console.WriteLine($"  row={idx} id={id} dim={dim} v1={av:G9} v2={bv:G9} Δ={av - bv:G6}");
        return 0;
    }
}
}
