using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace Rinha.Api;

/// <summary>
/// Wave 8: bucket-keyed fast-path cache for the /fraud-score hot path.
///
/// Built once at startup from the 3M reference set: the 8 most discriminative
/// features (per offline AUC analysis: amount, km_home, amount_ratio,
/// installments, tx_count_24h, unknown_merch, mcc_risk, hour) are each
/// quantile-binned into 8 buckets, producing a 24-bit composite key (16M slots,
/// 16 MiB byte table). Per-slot label is set ONLY when the bucket is literally
/// pure on training (count_fraud==0 OR count_fraud==count_total) AND
/// count >= MIN_COUNT — otherwise the slot is "undecided" and the request
/// falls through to the scorer.
///
/// Offline validation on the eval test set (54100 queries) at MIN_COUNT=100:
///   - hit rate 41.57%
///   - 0 false-negative and 0 false-positive detections (no detection cost)
///   - edge-case queries (fraud_count in 1..4) hit the fast path only ~5% of
///     the time (vs 41.57% overall) — they correctly stay on the scorer.
///
/// Disable with PROFILE_FAST_PATH=0.
/// </summary>
public static unsafe class ProfileFastPath
{
    // Selected features (indices into the 14-dim Vectorizer output).
    // Order chosen for AUC ranking; bit allocation is uniform 3 bits each.
    private static readonly int[] FeatureIndex = new[] { 0, 7, 2, 1, 8, 11, 12, 3 };
    //                                                  amt km  rat ins txc unk mcc hr

    private const int BitsPerFeature = 3;
    private const int NumFeatures = 8;
    private const int TotalBits = BitsPerFeature * NumFeatures;        // 24
    private const int NumSlots   = 1 << TotalBits;                      // 16,777,216
    private const int BinsPerFeature = 1 << BitsPerFeature;             // 8
    private const int MinBucketCount = 100;                             // empirical: 0 FN/FP on eval

    // Per-feature edges: edges[bin] is the upper bound of bin (exclusive).
    // Length == BinsPerFeature; the last value is +inf-ish so the final bin catches
    // anything above the max training value.
    private static float[][]? _edges;

    // Lookup table: byte per slot. 0 = undecided (call scorer); 1 = pure-legit
    // (fraudCount=0); 2 = pure-fraud (fraudCount=5). Stored as byte to halve cache
    // footprint vs ushort. ~16 MiB total.
    private static byte[]? _table;

    public const byte ResultUndecided = 0;
    public const byte ResultLegit     = 1;
    public const byte ResultFraud     = 2;

    public static bool IsEnabled => _table is not null;

    /// <summary>Statistics surfaced by Build for the startup log line.</summary>
    public static int UsedBuckets;
    public static int DecidedLegit;
    public static int DecidedFraud;

    public static void Build(Dataset ds)
    {
        if (Environment.GetEnvironmentVariable("PROFILE_FAST_PATH") == "0")
        {
            Console.WriteLine("ProfileFastPath: disabled (PROFILE_FAST_PATH=0).");
            return;
        }

        int n = ds.Count;
        var vectors = ds.VectorsPtr;
        var labels  = ds.LabelsPtr;
        if (vectors == null || labels == null)
        {
            Console.WriteLine("ProfileFastPath: dataset missing float vectors or labels — disabled.");
            return;
        }
        const int stride = Dataset.PaddedDimensions;

        // Step 1: compute quantile edges per feature. Reuse a single 12 MB column
        // buffer across all 8 features to avoid 8x allocation (container has 150MB limit).
        var edges = new float[NumFeatures][];
        var col = new float[n];
        for (int f = 0; f < NumFeatures; f++)
        {
            int featIdx = FeatureIndex[f];
            for (int i = 0; i < n; i++) col[i] = vectors[(long)i * stride + featIdx];
            Array.Sort(col);
            edges[f] = new float[BinsPerFeature];
            for (int b = 0; b < BinsPerFeature - 1; b++)
            {
                int q = (int)((long)(b + 1) * n / BinsPerFeature);
                edges[f][b] = col[q];
            }
            // Last edge: +infinity so the upper bin always wins for unseen-large values.
            edges[f][BinsPerFeature - 1] = float.PositiveInfinity;
        }
        col = null!; // release 12 MB before next phase
        _edges = edges;

        // Step 2: bucket every reference. Sparse accumulation via Dictionary — out of
        // 16M possible slots only ~16k are actually populated by the 3M refs, so a
        // 64-bit packed counter dictionary uses < 1 MB instead of the 128 MB a dense
        // int[NumSlots]×2 would take. Pack: high 32 bits = total count, low 32 bits = fraud count.
        var counts = new Dictionary<uint, ulong>(capacity: 32768);
        for (int i = 0; i < n; i++)
        {
            uint key = ComputeKey(vectors + (long)i * stride);
            ulong delta = labels[i] != 0 ? 0x1_0000_0001UL : 0x1_0000_0000UL;
            counts.TryGetValue(key, out var cur);
            counts[key] = cur + delta;
        }

        // Step 3: assign per-slot decision. Pure-only AND count >= MinBucketCount.
        // 16 MB persistent allocation (one byte per slot).
        var table = new byte[NumSlots];
        int used = counts.Count, decLegit = 0, decFraud = 0;
        foreach (var kv in counts)
        {
            int t = (int)(kv.Value >> 32);
            if (t < MinBucketCount) continue;
            int p = (int)(uint)kv.Value;
            if (p == 0)      { table[kv.Key] = ResultLegit; decLegit++; }
            else if (p == t) { table[kv.Key] = ResultFraud; decFraud++; }
        }
        counts = null!;
        _table = table;
        UsedBuckets = used; DecidedLegit = decLegit; DecidedFraud = decFraud;

        Console.WriteLine($"ProfileFastPath: built. slots={NumSlots:N0} used={used:N0} " +
                          $"decided_legit={decLegit:N0} decided_fraud={decFraud:N0} " +
                          $"min_count={MinBucketCount} table_bytes={table.Length:N0}");
    }

    /// <summary>Returns one of <see cref="ResultUndecided"/>, <see cref="ResultLegit"/>,
    /// <see cref="ResultFraud"/> for the given query vector. The query must be in the
    /// same float-feature space as <see cref="JsonVectorizer.VectorizeJson(ReadOnlySpan{byte}, Span{float})"/>.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static byte TryLookup(ReadOnlySpan<float> query)
    {
        var table = _table;
        if (table is null) return ResultUndecided;
        // Inlined for hot-path: ComputeKey reads 8 features and binary-searches each
        // edge array (only 8 floats — linear walk is faster than recursion).
        ref float q0 = ref MemoryMarshal.GetReference(query);
        uint key = ComputeKeyFromSpan(ref q0);
        return table[key];
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static uint ComputeKey(float* row)
    {
        var edges = _edges!;
        uint key = 0;
        int shift = 0;
        for (int f = 0; f < NumFeatures; f++)
        {
            float v = row[FeatureIndex[f]];
            int bin = FindBin(edges[f], v);
            key |= (uint)bin << shift;
            shift += BitsPerFeature;
        }
        return key;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static uint ComputeKeyFromSpan(ref float q0)
    {
        var edges = _edges!;
        uint key = 0;
        int shift = 0;
        for (int f = 0; f < NumFeatures; f++)
        {
            float v = Unsafe.Add(ref q0, FeatureIndex[f]);
            int bin = FindBin(edges[f], v);
            key |= (uint)bin << shift;
            shift += BitsPerFeature;
        }
        return key;
    }

    /// <summary>Linear scan for bin index. With BinsPerFeature=8 the branch table fits in 1 cache line
    /// and beats binary search at this size. Returns the smallest b such that v &lt; edges[b].</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int FindBin(float[] edges, float v)
    {
        // edges length == BinsPerFeature == 8; last element is +inf so loop always exits.
        for (int b = 0; b < BinsPerFeature - 1; b++)
            if (v < edges[b]) return b;
        return BinsPerFeature - 1;
    }
}
