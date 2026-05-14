using System.IO.MemoryMappedFiles;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace Rinha.Api;

/// <summary>
/// Memory-mapped, read-only access to the preprocessed reference dataset.
/// Vectors are stored as row-major float32 (Count * PaddedDimensions floats);
/// labels are stored as one byte per vector (0 = legit, 1 = fraud).
/// Optionally a quantized int8 view is also memory-mapped (Count * PaddedDimensions sbytes).
/// </summary>
public sealed unsafe class Dataset : IDisposable
{
    public const int Dimensions = 14;
    public const int PaddedDimensions = 16;
    public const float Q8Scale = 127f;
    public const float Q16Scale = 10000f;

    private readonly MemoryMappedFile _vectorsMmf;
    private readonly MemoryMappedViewAccessor _vectorsView;
    private readonly MemoryMappedFile _labelsMmf;
    private readonly MemoryMappedViewAccessor _labelsView;
    private readonly MemoryMappedFile? _q8Mmf;
    private readonly MemoryMappedViewAccessor? _q8View;
    private readonly MemoryMappedFile? _q8SoaMmf;
    private readonly MemoryMappedViewAccessor? _q8SoaView;
    private readonly MemoryMappedFile? _q16Mmf;
    private readonly MemoryMappedViewAccessor? _q16View;
    private readonly MemoryMappedFile? _q16SoaMmf;
    private readonly MemoryMappedViewAccessor? _q16SoaView;
    // Block-SoA Q16 layout: (TotalBlocks × 8 lanes × 14 dims) i16, dim-major within block.
    // Block b dim d lane l: ptr[(long)b * 112 + d * 8 + l].
    private readonly MemoryMappedFile? _q16BlockedMmf;
    private readonly MemoryMappedViewAccessor? _q16BlockedView;
    private readonly MemoryMappedFile? _blockOffsetsMmf;
    private readonly MemoryMappedViewAccessor? _blockOffsetsView;
    private readonly MemoryMappedFile? _centroidsMmf;
    private readonly MemoryMappedViewAccessor? _centroidsView;
    private readonly MemoryMappedFile? _offsetsMmf;
    private readonly MemoryMappedViewAccessor? _offsetsView;
    private readonly MemoryMappedFile? _bboxMinMmf;
    private readonly MemoryMappedViewAccessor? _bboxMinView;
    private readonly MemoryMappedFile? _bboxMaxMmf;
    private readonly MemoryMappedViewAccessor? _bboxMaxView;
    private readonly float* _vectorsPtr;
    private byte* _labelsPtr;
    private readonly sbyte* _q8Ptr;
    private readonly sbyte* _q8SoaPtr;
    private readonly short* _q16Ptr;
    private short* _q16BlockedPtr;
    private int* _blockOffsetsPtr;
    private float* _centroidsPtr;
    private int* _offsetsPtr;
    private float* _bboxMinPtr;
    private float* _bboxMaxPtr;
    // Column-major (SoA) transposed Q16 layout: Dimensions columns of Count shorts each.
    // Either pre-mmapped from a file (built at image-build time) or allocated lazily in
    // Prefetch() from the row-major Q16 mmap. Layout: _q16SoaPtr[d * Count + i].
    private short* _q16SoaPtr;
    private bool _q16SoaIsAllocated; // true = AlignedAlloc; false = mmap (do not free)

    public int Count { get; }
    public int NumCells { get; }
    public bool HasQ8 => _q8Ptr != null;
    public bool HasQ8Soa => _q8SoaPtr != null;
    public bool HasQ16 => _q16Ptr != null;
    public bool HasQ16Soa => _q16SoaPtr != null;
    public bool HasQ16Blocked => _q16BlockedPtr != null && _blockOffsetsPtr != null;
    public bool HasIvf => _centroidsPtr != null && _offsetsPtr != null;
    public bool HasIvfBbox => _bboxMinPtr != null && _bboxMaxPtr != null;

    private Dataset(
        MemoryMappedFile vectorsMmf, MemoryMappedViewAccessor vectorsView,
        MemoryMappedFile labelsMmf, MemoryMappedViewAccessor labelsView,
        MemoryMappedFile? q8Mmf, MemoryMappedViewAccessor? q8View,
        MemoryMappedFile? q8SoaMmf, MemoryMappedViewAccessor? q8SoaView,
        MemoryMappedFile? q16Mmf, MemoryMappedViewAccessor? q16View,
        MemoryMappedFile? q16SoaMmf, MemoryMappedViewAccessor? q16SoaView,
        MemoryMappedFile? q16BlockedMmf, MemoryMappedViewAccessor? q16BlockedView,
        MemoryMappedFile? blockOffsetsMmf, MemoryMappedViewAccessor? blockOffsetsView,
        MemoryMappedFile? centroidsMmf, MemoryMappedViewAccessor? centroidsView,
        MemoryMappedFile? offsetsMmf, MemoryMappedViewAccessor? offsetsView,
        MemoryMappedFile? bboxMinMmf, MemoryMappedViewAccessor? bboxMinView,
        MemoryMappedFile? bboxMaxMmf, MemoryMappedViewAccessor? bboxMaxView,
        int count, int numCells)
    {
        _vectorsMmf = vectorsMmf;
        _vectorsView = vectorsView;
        _labelsMmf = labelsMmf;
        _labelsView = labelsView;
        _q8Mmf = q8Mmf;
        _q8View = q8View;
        _q8SoaMmf = q8SoaMmf;
        _q8SoaView = q8SoaView;
        _q16Mmf = q16Mmf;
        _q16View = q16View;
        _q16SoaMmf = q16SoaMmf;
        _q16SoaView = q16SoaView;
        _q16BlockedMmf = q16BlockedMmf;
        _q16BlockedView = q16BlockedView;
        _blockOffsetsMmf = blockOffsetsMmf;
        _blockOffsetsView = blockOffsetsView;
        _centroidsMmf = centroidsMmf;
        _centroidsView = centroidsView;
        _offsetsMmf = offsetsMmf;
        _offsetsView = offsetsView;
        _bboxMinMmf = bboxMinMmf;
        _bboxMinView = bboxMinView;
        _bboxMaxMmf = bboxMaxMmf;
        _bboxMaxView = bboxMaxView;
        Count = count;
        NumCells = numCells;

        byte* basePtr = null;
        _vectorsView.SafeMemoryMappedViewHandle.AcquirePointer(ref basePtr);
        _vectorsPtr = (float*)basePtr;

        byte* labelsBase = null;
        _labelsView.SafeMemoryMappedViewHandle.AcquirePointer(ref labelsBase);
        _labelsPtr = labelsBase;

        if (_q8View is not null)
        {
            byte* q8Base = null;
            _q8View.SafeMemoryMappedViewHandle.AcquirePointer(ref q8Base);
            _q8Ptr = (sbyte*)q8Base;
        }

        if (_q8SoaView is not null)
        {
            byte* q8SoaBase = null;
            _q8SoaView.SafeMemoryMappedViewHandle.AcquirePointer(ref q8SoaBase);
            _q8SoaPtr = (sbyte*)q8SoaBase;
        }

        if (_q16View is not null)
        {
            byte* q16Base = null;
            _q16View.SafeMemoryMappedViewHandle.AcquirePointer(ref q16Base);
            _q16Ptr = (short*)q16Base;
        }

        if (_q16SoaView is not null)
        {
            byte* q16SoaBase = null;
            _q16SoaView.SafeMemoryMappedViewHandle.AcquirePointer(ref q16SoaBase);
            _q16SoaPtr = (short*)q16SoaBase;
        }

        if (_q16BlockedView is not null)
        {
            byte* qbBase = null;
            _q16BlockedView.SafeMemoryMappedViewHandle.AcquirePointer(ref qbBase);
            _q16BlockedPtr = (short*)qbBase;
        }

        if (_blockOffsetsView is not null)
        {
            byte* boBase = null;
            _blockOffsetsView.SafeMemoryMappedViewHandle.AcquirePointer(ref boBase);
            _blockOffsetsPtr = (int*)boBase;
        }

        if (_centroidsView is not null)
        {
            byte* cBase = null;
            _centroidsView.SafeMemoryMappedViewHandle.AcquirePointer(ref cBase);
            _centroidsPtr = (float*)cBase;
        }

        if (_offsetsView is not null)
        {
            byte* oBase = null;
            _offsetsView.SafeMemoryMappedViewHandle.AcquirePointer(ref oBase);
            _offsetsPtr = (int*)oBase;
        }

        if (_bboxMinView is not null)
        {
            byte* bMinBase = null;
            _bboxMinView.SafeMemoryMappedViewHandle.AcquirePointer(ref bMinBase);
            _bboxMinPtr = (float*)bMinBase;
        }

        if (_bboxMaxView is not null)
        {
            byte* bMaxBase = null;
            _bboxMaxView.SafeMemoryMappedViewHandle.AcquirePointer(ref bMaxBase);
            _bboxMaxPtr = (float*)bMaxBase;
        }
    }

    /// <summary>
    /// Pre-touch all mmap-backed pages so the first user requests don't pay
    /// minor page-fault overhead. Reads one byte per 4KB page across every
    /// view that was actually opened. Returns the total bytes scanned.
    /// L3: also calls madvise(MADV_HUGEPAGE) on each region to promote 4KB
    /// pages to 2MB hugepages, shrinking the TLB footprint dramatically
    /// (192MB float dataset = 48k 4KB pages → 96 2MB pages).
    /// </summary>
    /// <summary>
    /// Pre-touch mmap pages for the active scorer to eliminate first-request page-fault overhead.
    /// scorerHint controls which regions are warmed:
    ///   "brute" → Q16-SoA only (~84MB). Float32/Q8/IVF files are left demand-paged.
    ///   other   → Q8, Q8-SoA, IVF centroids/offsets/bbox (~90MB). Q16-SoA left demand-paged.
    /// Float32 vectors (192MB) are never prefetched — opt-in via PREFETCH_FLOAT=1.
    /// </summary>
    public long Prefetch(string? scorerHint = null)
    {
        const int PageSize = 4096;
        long total = 0;
        long sink = 0;
        long hpTotal = 0;
        long mlTotal = 0;
        bool isBrute = scorerHint == "brute";
        bool isBlocked = scorerHint == "ivf-blocked";

        // Float32 vectors (192MB): not needed by IVF or brute-Q16 scorer; skip by default.
        if (Environment.GetEnvironmentVariable("PREFETCH_FLOAT") == "1" && _vectorsPtr != null)
        {
            long vBytes = (long)Count * PaddedDimensions * sizeof(float);
            hpTotal += AdviseHuge((byte*)_vectorsPtr, vBytes);
            sink += TouchPages((byte*)_vectorsPtr, vBytes, PageSize);
            total += vBytes;
        }

        // Labels: tiny, always prefetch.
        long lBytes = Count;
        hpTotal += AdviseHuge(_labelsPtr, lBytes);
        sink += TouchPages(_labelsPtr, lBytes, PageSize);
        mlTotal += MlockRegion(_labelsPtr, lBytes);
        total += lBytes;

        // Q8/Q8-SoA: needed by IVF (Q8) scorer, not by brute or ivf-blocked.
        if (!isBrute && !isBlocked)
        {
            if (_q8Ptr != null)
            {
                long bytes = (long)Count * 16;
                hpTotal += AdviseHuge((byte*)_q8Ptr, bytes);
                sink += TouchPages((byte*)_q8Ptr, bytes, PageSize);
                mlTotal += MlockRegion((byte*)_q8Ptr, bytes);
                total += bytes;
            }
            if (_q8SoaPtr != null)
            {
                long bytes = (long)Count * Dimensions;
                hpTotal += AdviseHuge((byte*)_q8SoaPtr, bytes);
                sink += TouchPages((byte*)_q8SoaPtr, bytes, PageSize);
                total += bytes;
            }
        }

        // Centroids/offsets/bbox: needed by both Q8 IVF and ivf-blocked.
        if (!isBrute)
        {
            if (_centroidsPtr != null)
            {
                long bytes = (long)NumCells * PaddedDimensions * sizeof(float);
                hpTotal += AdviseHuge((byte*)_centroidsPtr, bytes);
                sink += TouchPages((byte*)_centroidsPtr, bytes, PageSize);
                mlTotal += MlockRegion((byte*)_centroidsPtr, bytes);
                total += bytes;
            }
            if (_offsetsPtr != null)
            {
                long bytes = (long)(NumCells + 1) * sizeof(int);
                hpTotal += AdviseHuge((byte*)_offsetsPtr, bytes);
                sink += TouchPages((byte*)_offsetsPtr, bytes, PageSize);
                mlTotal += MlockRegion((byte*)_offsetsPtr, bytes);
                total += bytes;
            }
            if (_bboxMinPtr != null)
            {
                long bytes = (long)NumCells * PaddedDimensions * sizeof(float);
                hpTotal += AdviseHuge((byte*)_bboxMinPtr, bytes);
                sink += TouchPages((byte*)_bboxMinPtr, bytes, PageSize);
                total += bytes;
            }
            if (_bboxMaxPtr != null)
            {
                long bytes = (long)NumCells * PaddedDimensions * sizeof(float);
                hpTotal += AdviseHuge((byte*)_bboxMaxPtr, bytes);
                sink += TouchPages((byte*)_bboxMaxPtr, bytes, PageSize);
                total += bytes;
            }
        }

        // Block-SoA Q16 + block_offsets: needed only by ivf-blocked.
        if (isBlocked && _q16BlockedPtr != null && _blockOffsetsPtr != null)
        {
            long boBytes = (long)(NumCells + 1) * sizeof(int);
            hpTotal += AdviseHuge((byte*)_blockOffsetsPtr, boBytes);
            sink += TouchPages((byte*)_blockOffsetsPtr, boBytes, PageSize);
            total += boBytes;

            // Total blocks = block_offsets[NumCells]; read directly to size the warm.
            int totalBlocks = _blockOffsetsPtr[NumCells];
            long bbBytes = (long)totalBlocks * 8 * Dimensions * sizeof(short);
            hpTotal += AdviseHuge((byte*)_q16BlockedPtr, bbBytes);
            sink += TouchPages((byte*)_q16BlockedPtr, bbBytes, PageSize);
            mlTotal += MlockRegion((byte*)_q16BlockedPtr, bbBytes);
            total += bbBytes;
        }

        // Q16-SoA (column-major): needed by brute-force scorer only.
        // If pre-mmapped from a file, warm up pages. Otherwise transpose from Q16 AoS.
        long soaBytes = (long)Dimensions * Count * sizeof(short);
        if (isBrute && _q16SoaPtr != null)
        {
            hpTotal += AdviseHuge((byte*)_q16SoaPtr, soaBytes);
            sink += TouchPages((byte*)_q16SoaPtr, soaBytes, PageSize);
            mlTotal += MlockRegion((byte*)_q16SoaPtr, soaBytes);
            total += soaBytes;
        }
        else if (isBrute && _q16Ptr != null)
        {
            // Fallback: allocate and transpose from Q16 AoS (tiled to limit peak RSS).
            _q16SoaPtr = (short*)NativeMemory.AlignedAlloc((nuint)soaBytes, 64);
            _q16SoaIsAllocated = true;
            short* src = _q16Ptr;
            short* dst = _q16SoaPtr;
            int count = Count;
            int paddedDims = PaddedDimensions;
            const int TileRows = 16384;
            const int MADV_DONTNEED = 4;
            for (int tileStart = 0; tileStart < count; tileStart += TileRows)
            {
                int tileEnd = Math.Min(tileStart + TileRows, count);
                for (int i = tileStart; i < tileEnd; i++)
                {
                    short* row = src + (long)i * paddedDims;
                    for (int d = 0; d < Dimensions; d++)
                        dst[(long)d * count + i] = row[d];
                }
                if (OperatingSystem.IsLinux())
                {
                    byte* tileBase = (byte*)(src + (long)tileStart * paddedDims);
                    int tileLen = tileEnd - tileStart;
                    long tileBytes = (long)tileLen * paddedDims * sizeof(short);
                    LinuxMadvise((IntPtr)tileBase, (UIntPtr)tileBytes, MADV_DONTNEED);
                }
            }
            hpTotal += AdviseHuge((byte*)_q16SoaPtr, soaBytes);
            sink += TouchPages((byte*)_q16SoaPtr, soaBytes, PageSize);
            mlTotal += MlockRegion((byte*)_q16SoaPtr, soaBytes);
            total += soaBytes;
        }

        GC.KeepAlive(sink);
        LastHugepageAdvisedBytes = hpTotal;
        LastMlockedBytes = mlTotal;
        if (s_hugeAnonEnabled)
            LastAnonHugeBytes = PromoteHotPathToAnonHuge(scorerHint);
        return total;
    }

    /// <summary>L4: replaces file-backed (r--s) mmap pointers for the active scorer's hot
    /// data with anonymous-mapped copies eligible for THP collapse. Targets the regions that
    /// are randomly accessed during scoring (where TLB misses hurt most). Returns total bytes
    /// successfully promoted; on partial failure the original mmap pointer is kept.</summary>
    private long PromoteHotPathToAnonHuge(string? scorerHint)
    {
        long promoted = 0;
        bool isBrute = scorerHint == "brute";
        bool isBlocked = scorerHint == "ivf-blocked";

        // labels (always touched on top-K insert) — small but hot.
        if (_labelsPtr != null)
        {
            byte* anon = CopyToHugeAnon(_labelsPtr, Count);
            if (anon != null) { _labelsPtr = anon; promoted += Count; }
        }

        // ivf-blocked hot path: q16_blocked + block_offsets + centroids + bbox_min/max.
        if (isBlocked)
        {
            if (_blockOffsetsPtr != null)
            {
                long boBytes = (long)(NumCells + 1) * sizeof(int);
                byte* anon = CopyToHugeAnon((byte*)_blockOffsetsPtr, boBytes);
                if (anon != null) { _blockOffsetsPtr = (int*)anon; promoted += boBytes; }
            }
            if (_q16BlockedPtr != null && _blockOffsetsPtr != null)
            {
                int totalBlocks = _blockOffsetsPtr[NumCells];
                long bbBytes = (long)totalBlocks * 8 * Dimensions * sizeof(short);
                byte* anon = CopyToHugeAnon((byte*)_q16BlockedPtr, bbBytes);
                if (anon != null) { _q16BlockedPtr = (short*)anon; promoted += bbBytes; }
            }
            if (_centroidsPtr != null)
            {
                long bytes = (long)NumCells * PaddedDimensions * sizeof(float);
                byte* anon = CopyToHugeAnon((byte*)_centroidsPtr, bytes);
                if (anon != null) { _centroidsPtr = (float*)anon; promoted += bytes; }
            }
            if (_offsetsPtr != null)
            {
                long bytes = (long)(NumCells + 1) * sizeof(int);
                byte* anon = CopyToHugeAnon((byte*)_offsetsPtr, bytes);
                if (anon != null) { _offsetsPtr = (int*)anon; promoted += bytes; }
            }
            if (_bboxMinPtr != null)
            {
                long bytes = (long)NumCells * PaddedDimensions * sizeof(float);
                byte* anon = CopyToHugeAnon((byte*)_bboxMinPtr, bytes);
                if (anon != null) { _bboxMinPtr = (float*)anon; promoted += bytes; }
            }
            if (_bboxMaxPtr != null)
            {
                long bytes = (long)NumCells * PaddedDimensions * sizeof(float);
                byte* anon = CopyToHugeAnon((byte*)_bboxMaxPtr, bytes);
                if (anon != null) { _bboxMaxPtr = (float*)anon; promoted += bytes; }
            }
        }

        // Brute-force hot path: q16_soa.
        if (isBrute && _q16SoaPtr != null)
        {
            long soaBytes = (long)Dimensions * Count * sizeof(short);
            byte* anon = CopyToHugeAnon((byte*)_q16SoaPtr, soaBytes);
            if (anon != null) { _q16SoaPtr = (short*)anon; promoted += soaBytes; }
        }

        return promoted;
    }

    /// <summary>Bytes successfully copied into hugepage-eligible anon memory on the last
    /// Prefetch call. 0 on non-Linux or when DATASET_HUGE_ANON != 1.</summary>
    public long LastAnonHugeBytes { get; private set; }

    /// <summary>Total bytes successfully advised with MADV_HUGEPAGE on the last Prefetch call. 0 on non-Linux or if disabled.</summary>
    public long LastHugepageAdvisedBytes { get; private set; }

    [DllImport("libc", EntryPoint = "madvise", SetLastError = true)]
    private static extern int LinuxMadvise(IntPtr addr, UIntPtr length, int advice);
    [DllImport("libc", EntryPoint = "mlock", SetLastError = true)]
    private static extern int LinuxMlock(IntPtr addr, UIntPtr length);
    [DllImport("libc", EntryPoint = "mmap", SetLastError = true)]
    private static extern IntPtr LinuxMmap(IntPtr addr, UIntPtr length, int prot, int flags, int fd, IntPtr offset);
    [DllImport("libc", EntryPoint = "munmap", SetLastError = true)]
    private static extern int LinuxMunmap(IntPtr addr, UIntPtr length);
    private const int MADV_HUGEPAGE = 14;
    private const int MADV_DONTNEED = 4;
    private const int PROT_READ = 1, PROT_WRITE = 2;
    private const int MAP_PRIVATE = 0x02, MAP_ANONYMOUS = 0x20;
    private const long HugePageSize = 2L * 1024 * 1024;
    private static readonly IntPtr MAP_FAILED = new IntPtr(-1);
    private static readonly bool s_thpEnabled = OperatingSystem.IsLinux();
    private static readonly bool s_mlockEnabled = false;
    private static readonly bool s_hugeAnonEnabled =
        OperatingSystem.IsLinux() && Environment.GetEnvironmentVariable("DATASET_HUGE_ANON") == "1";

    /// <summary>L4: allocate a 2MB-aligned anonymous mapping and call madvise(MADV_HUGEPAGE)
    /// so khugepaged can collapse the 4KB pages into 2MB hugepages. Returns null on failure.
    /// File-backed shared mmaps (r--s) are silently ignored by THP on most kernels because
    /// READ_ONLY_THP_FOR_FS isn't enabled — copying the dataset bytes into anon memory is
    /// the only portable way to actually get hugepages for our reference vectors.</summary>
    private static byte* MmapAnonHugeAligned(long bytes)
    {
        if (!OperatingSystem.IsLinux() || bytes <= 0) return null;
        // Over-allocate by HugePageSize to guarantee we can find a 2MB-aligned start.
        long total = bytes + HugePageSize;
        IntPtr raw = LinuxMmap(IntPtr.Zero, (UIntPtr)total, PROT_READ | PROT_WRITE,
            MAP_PRIVATE | MAP_ANONYMOUS, -1, IntPtr.Zero);
        if (raw == MAP_FAILED) return null;
        long rawAddr = raw.ToInt64();
        long alignedAddr = (rawAddr + HugePageSize - 1) & ~(HugePageSize - 1);
        long headWaste = alignedAddr - rawAddr;
        long tailStart = alignedAddr + bytes;
        long tailWaste = (rawAddr + total) - tailStart;
        // Trim the unaligned head and tail back to the kernel.
        if (headWaste > 0) LinuxMunmap(raw, (UIntPtr)headWaste);
        if (tailWaste > 0) LinuxMunmap(new IntPtr(tailStart), (UIntPtr)tailWaste);
        // Hint THP on the aligned region so khugepaged collapses 4KB → 2MB.
        LinuxMadvise(new IntPtr(alignedAddr), (UIntPtr)bytes, MADV_HUGEPAGE);
        return (byte*)alignedAddr;
    }

    /// <summary>L4: copy <paramref name="bytes"/> from <paramref name="src"/> into a fresh
    /// hugepage-eligible anonymous region. Returns null on failure (caller keeps the original
    /// mmap pointer).</summary>
    private static byte* CopyToHugeAnon(byte* src, long bytes)
    {
        byte* dst = MmapAnonHugeAligned(bytes);
        if (dst == null) return null;
        Buffer.MemoryCopy(src, dst, bytes, bytes);
        return dst;
    }

    private static long AdviseHuge(byte* p, long bytes)
    {
        if (!s_thpEnabled || bytes <= 0) return 0;
        // madvise requires page-aligned start; mmap base is always page-aligned,
        // so just round length up to whole pages.
        UIntPtr len = (UIntPtr)bytes;
        try
        {
            int rc = LinuxMadvise((IntPtr)p, len, MADV_HUGEPAGE);
            return rc == 0 ? bytes : 0;
        }
        catch
        {
            return 0;
        }
    }

    /// <summary>L2-cascade follow-up: mlock() the dataset to prevent the kernel
    /// from evicting pages when memory traffic drops (e.g., when cascade fast-paths
    /// most queries away from Q8 scans). Returns bytes successfully locked.</summary>
    public long LastMlockedBytes { get; private set; }

    private static long MlockRegion(byte* p, long bytes)
    {
        if (!s_mlockEnabled || bytes <= 0) return 0;
        try
        {
            int rc = LinuxMlock((IntPtr)p, (UIntPtr)bytes);
            return rc == 0 ? bytes : 0;
        }
        catch
        {
            return 0;
        }
    }

    private static long TouchPages(byte* p, long bytes, int pageSize)
    {
        long acc = 0;
        for (long i = 0; i < bytes; i += pageSize) acc += p[i];
        if (bytes > 0) acc += p[bytes - 1];
        return acc;
    }

    public static Dataset Open(
        string vectorsPath,
        string labelsPath,
        string? vectorsQ8Path = null,
        string? vectorsQ8SoaPath = null,
        string? vectorsQ16Path = null,
        string? vectorsQ16SoaPath = null,
        string? ivfCentroidsPath = null,
        string? ivfOffsetsPath = null,
        string? ivfBboxMinPath = null,
        string? ivfBboxMaxPath = null,
        string? vectorsQ16BlockedPath = null,
        string? ivfBlockOffsetsPath = null)
    {
        var vectorsLen = new FileInfo(vectorsPath).Length;
        var labelsLen = new FileInfo(labelsPath).Length;
        long bytesPerRow = PaddedDimensions * sizeof(float);
        if (vectorsLen % bytesPerRow != 0)
            throw new InvalidDataException($"Vectors file size {vectorsLen} not multiple of row size {bytesPerRow}");
        long count = vectorsLen / bytesPerRow;
        if (labelsLen != count)
            throw new InvalidDataException($"Labels count {labelsLen} != vectors count {count}");
        if (count > int.MaxValue)
            throw new InvalidDataException("Dataset too large");

        var vectorsMmf = MemoryMappedFile.CreateFromFile(vectorsPath, FileMode.Open, mapName: null, capacity: 0, MemoryMappedFileAccess.Read);
        var vectorsView = vectorsMmf.CreateViewAccessor(0, 0, MemoryMappedFileAccess.Read);

        var labelsMmf = MemoryMappedFile.CreateFromFile(labelsPath, FileMode.Open, mapName: null, capacity: 0, MemoryMappedFileAccess.Read);
        var labelsView = labelsMmf.CreateViewAccessor(0, 0, MemoryMappedFileAccess.Read);

        MemoryMappedFile? q8Mmf = null;
        MemoryMappedViewAccessor? q8View = null;
        if (!string.IsNullOrEmpty(vectorsQ8Path) && File.Exists(vectorsQ8Path))
        {
            var q8Len = new FileInfo(vectorsQ8Path).Length;
            long q8Expected = count * PaddedDimensions;
            if (q8Len != q8Expected)
                throw new InvalidDataException($"Q8 file size {q8Len} != expected {q8Expected}");
            q8Mmf = MemoryMappedFile.CreateFromFile(vectorsQ8Path, FileMode.Open, mapName: null, capacity: 0, MemoryMappedFileAccess.Read);
            q8View = q8Mmf.CreateViewAccessor(0, 0, MemoryMappedFileAccess.Read);
        }

        // J11: SoA layout (14 contiguous N-byte blocks). Cache-friendly for per-dim
        // streaming with row-survivor compaction (scalar early-abort).
        MemoryMappedFile? q8SoaMmf = null;
        MemoryMappedViewAccessor? q8SoaView = null;
        if (!string.IsNullOrEmpty(vectorsQ8SoaPath) && File.Exists(vectorsQ8SoaPath))
        {
            var soaLen = new FileInfo(vectorsQ8SoaPath).Length;
            long soaExpected = count * Dimensions;
            if (soaLen != soaExpected)
                throw new InvalidDataException($"Q8-SoA file size {soaLen} != expected {soaExpected}");
            q8SoaMmf = MemoryMappedFile.CreateFromFile(vectorsQ8SoaPath, FileMode.Open, mapName: null, capacity: 0, MemoryMappedFileAccess.Read);
            q8SoaView = q8SoaMmf.CreateViewAccessor(0, 0, MemoryMappedFileAccess.Read);
        }

        // J25: Q16 (int16) layout for high-precision rerank — replaces float on hot path.
        MemoryMappedFile? q16Mmf = null;
        MemoryMappedViewAccessor? q16View = null;
        if (!string.IsNullOrEmpty(vectorsQ16Path) && File.Exists(vectorsQ16Path))
        {
            var q16Len = new FileInfo(vectorsQ16Path).Length;
            long q16Expected = count * PaddedDimensions * sizeof(short);
            if (q16Len != q16Expected)
                throw new InvalidDataException($"Q16 file size {q16Len} != expected {q16Expected}");
            q16Mmf = MemoryMappedFile.CreateFromFile(vectorsQ16Path, FileMode.Open, mapName: null, capacity: 0, MemoryMappedFileAccess.Read);
            q16View = q16Mmf.CreateViewAccessor(0, 0, MemoryMappedFileAccess.Read);
        }

        MemoryMappedFile? q16SoaMmf = null;
        MemoryMappedViewAccessor? q16SoaView = null;
        if (!string.IsNullOrEmpty(vectorsQ16SoaPath) && File.Exists(vectorsQ16SoaPath))
        {
            var q16SoaLen = new FileInfo(vectorsQ16SoaPath).Length;
            long q16SoaExpected = count * Dimensions * sizeof(short);
            if (q16SoaLen != q16SoaExpected)
                throw new InvalidDataException($"Q16-SoA file size {q16SoaLen} != expected {q16SoaExpected}");
            q16SoaMmf = MemoryMappedFile.CreateFromFile(vectorsQ16SoaPath, FileMode.Open, mapName: null, capacity: 0, MemoryMappedFileAccess.Read);
            q16SoaView = q16SoaMmf.CreateViewAccessor(0, 0, MemoryMappedFileAccess.Read);
        }

        // Block-SoA Q16: built by Rinha.Preprocessor --block; layout 8 lanes × 14 dims = 112 i16/block.
        // Padding lanes filled with short.MaxValue so they always lose top-K race.
        MemoryMappedFile? q16BlockedMmf = null;
        MemoryMappedViewAccessor? q16BlockedView = null;
        MemoryMappedFile? blockOffsetsMmf = null;
        MemoryMappedViewAccessor? blockOffsetsView = null;
        if (!string.IsNullOrEmpty(vectorsQ16BlockedPath) && File.Exists(vectorsQ16BlockedPath)
            && !string.IsNullOrEmpty(ivfBlockOffsetsPath) && File.Exists(ivfBlockOffsetsPath))
        {
            var bLen = new FileInfo(vectorsQ16BlockedPath).Length;
            long perBlock = (long)8 * Dimensions * sizeof(short); // 224 bytes
            if (bLen % perBlock != 0)
                throw new InvalidDataException($"Q16-blocked size {bLen} not multiple of {perBlock}");
            var boLen = new FileInfo(ivfBlockOffsetsPath).Length;
            if (boLen % sizeof(int) != 0)
                throw new InvalidDataException($"Block-offsets size {boLen} not multiple of 4");
            q16BlockedMmf = MemoryMappedFile.CreateFromFile(vectorsQ16BlockedPath, FileMode.Open, null, 0, MemoryMappedFileAccess.Read);
            q16BlockedView = q16BlockedMmf.CreateViewAccessor(0, 0, MemoryMappedFileAccess.Read);
            blockOffsetsMmf = MemoryMappedFile.CreateFromFile(ivfBlockOffsetsPath, FileMode.Open, null, 0, MemoryMappedFileAccess.Read);
            blockOffsetsView = blockOffsetsMmf.CreateViewAccessor(0, 0, MemoryMappedFileAccess.Read);
        }

        MemoryMappedFile? centroidsMmf = null;
        MemoryMappedViewAccessor? centroidsView = null;
        MemoryMappedFile? offsetsMmf = null;
        MemoryMappedViewAccessor? offsetsView = null;
        int numCells = 0;
        if (!string.IsNullOrEmpty(ivfCentroidsPath) && File.Exists(ivfCentroidsPath)
            && !string.IsNullOrEmpty(ivfOffsetsPath) && File.Exists(ivfOffsetsPath))
        {
            var cLen = new FileInfo(ivfCentroidsPath).Length;
            long perCentroid = PaddedDimensions * sizeof(float);
            if (cLen % perCentroid != 0)
                throw new InvalidDataException($"Centroids size {cLen} not multiple of {perCentroid}");
            long nlist = cLen / perCentroid;
            var oLen = new FileInfo(ivfOffsetsPath).Length;
            if (oLen != (nlist + 1) * sizeof(int))
                throw new InvalidDataException($"Offsets size {oLen} != expected {(nlist + 1) * sizeof(int)}");
            numCells = (int)nlist;

            centroidsMmf = MemoryMappedFile.CreateFromFile(ivfCentroidsPath, FileMode.Open, null, 0, MemoryMappedFileAccess.Read);
            centroidsView = centroidsMmf.CreateViewAccessor(0, 0, MemoryMappedFileAccess.Read);
            offsetsMmf = MemoryMappedFile.CreateFromFile(ivfOffsetsPath, FileMode.Open, null, 0, MemoryMappedFileAccess.Read);
            offsetsView = offsetsMmf.CreateViewAccessor(0, 0, MemoryMappedFileAccess.Read);
        }

        MemoryMappedFile? bboxMinMmf = null;
        MemoryMappedViewAccessor? bboxMinView = null;
        MemoryMappedFile? bboxMaxMmf = null;
        MemoryMappedViewAccessor? bboxMaxView = null;
        if (numCells > 0
            && !string.IsNullOrEmpty(ivfBboxMinPath) && File.Exists(ivfBboxMinPath)
            && !string.IsNullOrEmpty(ivfBboxMaxPath) && File.Exists(ivfBboxMaxPath))
        {
            long expected = (long)numCells * PaddedDimensions * sizeof(float);
            long mLen = new FileInfo(ivfBboxMinPath).Length;
            long xLen = new FileInfo(ivfBboxMaxPath).Length;
            if (mLen != expected) throw new InvalidDataException($"BboxMin size {mLen} != expected {expected}");
            if (xLen != expected) throw new InvalidDataException($"BboxMax size {xLen} != expected {expected}");
            bboxMinMmf = MemoryMappedFile.CreateFromFile(ivfBboxMinPath, FileMode.Open, null, 0, MemoryMappedFileAccess.Read);
            bboxMinView = bboxMinMmf.CreateViewAccessor(0, 0, MemoryMappedFileAccess.Read);
            bboxMaxMmf = MemoryMappedFile.CreateFromFile(ivfBboxMaxPath, FileMode.Open, null, 0, MemoryMappedFileAccess.Read);
            bboxMaxView = bboxMaxMmf.CreateViewAccessor(0, 0, MemoryMappedFileAccess.Read);
        }

        return new Dataset(
            vectorsMmf, vectorsView, labelsMmf, labelsView,
            q8Mmf, q8View, q8SoaMmf, q8SoaView,
            q16Mmf, q16View, q16SoaMmf, q16SoaView,
            q16BlockedMmf, q16BlockedView, blockOffsetsMmf, blockOffsetsView,
            centroidsMmf, centroidsView, offsetsMmf, offsetsView,
            bboxMinMmf, bboxMinView, bboxMaxMmf, bboxMaxView,
            (int)count, numCells);
    }

    public float* VectorsPtr
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => _vectorsPtr;
    }

    public byte* LabelsPtr
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => _labelsPtr;
    }

    public sbyte* Q8VectorsPtr
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => _q8Ptr;
    }

    /// <summary>J11: returns a pointer to the SoA Q8 block — concatenated 14 N-byte arrays.
    /// Dim d's row r value is at offset (long)d * Count + r.</summary>
    public sbyte* Q8SoaPtr
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => _q8SoaPtr;
    }

    /// <summary>J25: int16 dense AoS reference vectors (16 shorts per row, padded).
    /// Same row order as Q8VectorsPtr (post-IVF reorder). Scale = Q16Scale (10000).</summary>
    public short* Q16VectorsPtr
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => _q16Ptr;
    }

    /// <summary>Column-major (SoA) transposed Q16 layout built in Prefetch().
    /// Dim d's column: &amp;Q16SoaPtr[d * Count]. Populated only if HasQ16Soa is true.</summary>
    public short* Q16SoaPtr
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => _q16SoaPtr;
    }

    /// <summary>Block-SoA Q16 layout (8 lanes × 14 dims = 112 i16 per block, dim-major within block).
    /// Used by IvfBlockedScorer. Populated only if HasQ16Blocked is true.</summary>
    public short* Q16BlockedPtr
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => _q16BlockedPtr;
    }

    /// <summary>Per-cell prefix sum of block counts (length NumCells+1). Cell c covers
    /// blocks [BlockOffsetsPtr[c], BlockOffsetsPtr[c+1]). Populated only if HasQ16Blocked is true.</summary>
    public int* BlockOffsetsPtr
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => _blockOffsetsPtr;
    }

    public float* CentroidsPtr
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => _centroidsPtr;
    }

    public int* CellOffsetsPtr
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => _offsetsPtr;
    }

    public float* IvfBboxMinPtr
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => _bboxMinPtr;
    }

    public float* IvfBboxMaxPtr
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => _bboxMaxPtr;
    }

    public void Dispose()
    {
        try { _vectorsView.SafeMemoryMappedViewHandle.ReleasePointer(); } catch { }
        try { _labelsView.SafeMemoryMappedViewHandle.ReleasePointer(); } catch { }
        try { _q8View?.SafeMemoryMappedViewHandle.ReleasePointer(); } catch { }
        try { _q8SoaView?.SafeMemoryMappedViewHandle.ReleasePointer(); } catch { }
        try { _q16View?.SafeMemoryMappedViewHandle.ReleasePointer(); } catch { }
        try { _q16SoaView?.SafeMemoryMappedViewHandle.ReleasePointer(); } catch { }
        try { _q16BlockedView?.SafeMemoryMappedViewHandle.ReleasePointer(); } catch { }
        try { _blockOffsetsView?.SafeMemoryMappedViewHandle.ReleasePointer(); } catch { }
        try { _centroidsView?.SafeMemoryMappedViewHandle.ReleasePointer(); } catch { }
        try { _offsetsView?.SafeMemoryMappedViewHandle.ReleasePointer(); } catch { }
        try { _bboxMinView?.SafeMemoryMappedViewHandle.ReleasePointer(); } catch { }
        try { _bboxMaxView?.SafeMemoryMappedViewHandle.ReleasePointer(); } catch { }
        if (_q16SoaPtr != null && _q16SoaIsAllocated)
        {
            NativeMemory.AlignedFree(_q16SoaPtr);
            _q16SoaPtr = null;
        }
        _vectorsView.Dispose();
        _vectorsMmf.Dispose();
        _labelsView.Dispose();
        _labelsMmf.Dispose();
        _q8View?.Dispose();
        _q8Mmf?.Dispose();
        _q8SoaView?.Dispose();
        _q8SoaMmf?.Dispose();
        _q16View?.Dispose();
        _q16Mmf?.Dispose();
        _q16SoaView?.Dispose();
        _q16SoaMmf?.Dispose();
        _q16BlockedView?.Dispose();
        _q16BlockedMmf?.Dispose();
        _blockOffsetsView?.Dispose();
        _blockOffsetsMmf?.Dispose();
        _centroidsView?.Dispose();
        _centroidsMmf?.Dispose();
        _offsetsView?.Dispose();
        _offsetsMmf?.Dispose();
        _bboxMinView?.Dispose();
        _bboxMinMmf?.Dispose();
        _bboxMaxView?.Dispose();
        _bboxMaxMmf?.Dispose();
    }
}
