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
    private readonly MemoryMappedFile? _centroidsMmf;
    private readonly MemoryMappedViewAccessor? _centroidsView;
    private readonly MemoryMappedFile? _offsetsMmf;
    private readonly MemoryMappedViewAccessor? _offsetsView;
    private readonly MemoryMappedFile? _bboxMinMmf;
    private readonly MemoryMappedViewAccessor? _bboxMinView;
    private readonly MemoryMappedFile? _bboxMaxMmf;
    private readonly MemoryMappedViewAccessor? _bboxMaxView;
    private readonly float* _vectorsPtr;
    private readonly byte* _labelsPtr;
    private readonly sbyte* _q8Ptr;
    private readonly sbyte* _q8SoaPtr;
    private readonly short* _q16Ptr;
    private readonly float* _centroidsPtr;
    private readonly int* _offsetsPtr;
    private readonly float* _bboxMinPtr;
    private readonly float* _bboxMaxPtr;
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
    public bool HasIvf => _centroidsPtr != null && _offsetsPtr != null;
    public bool HasIvfBbox => _bboxMinPtr != null && _bboxMaxPtr != null;

    private Dataset(
        MemoryMappedFile vectorsMmf, MemoryMappedViewAccessor vectorsView,
        MemoryMappedFile labelsMmf, MemoryMappedViewAccessor labelsView,
        MemoryMappedFile? q8Mmf, MemoryMappedViewAccessor? q8View,
        MemoryMappedFile? q8SoaMmf, MemoryMappedViewAccessor? q8SoaView,
        MemoryMappedFile? q16Mmf, MemoryMappedViewAccessor? q16View,
        MemoryMappedFile? q16SoaMmf, MemoryMappedViewAccessor? q16SoaView,
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

        // Q8/Q8-SoA/IVF: needed by IVF scorer, not by brute-Q16.
        if (!isBrute)
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
        return total;
    }

    /// <summary>Total bytes successfully advised with MADV_HUGEPAGE on the last Prefetch call. 0 on non-Linux or if disabled.</summary>
    public long LastHugepageAdvisedBytes { get; private set; }

    [DllImport("libc", EntryPoint = "madvise", SetLastError = true)]
    private static extern int LinuxMadvise(IntPtr addr, UIntPtr length, int advice);
    [DllImport("libc", EntryPoint = "mlock", SetLastError = true)]
    private static extern int LinuxMlock(IntPtr addr, UIntPtr length);
    private const int MADV_HUGEPAGE = 14;
    private static readonly bool s_thpEnabled = OperatingSystem.IsLinux();
    private static readonly bool s_mlockEnabled = false;

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
        string? ivfBboxMaxPath = null)
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
