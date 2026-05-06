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

    private readonly MemoryMappedFile _vectorsMmf;
    private readonly MemoryMappedViewAccessor _vectorsView;
    private readonly MemoryMappedFile _labelsMmf;
    private readonly MemoryMappedViewAccessor _labelsView;
    private readonly MemoryMappedFile? _q8Mmf;
    private readonly MemoryMappedViewAccessor? _q8View;
    private readonly MemoryMappedFile? _q8SoaMmf;
    private readonly MemoryMappedViewAccessor? _q8SoaView;
    private readonly MemoryMappedFile? _centroidsMmf;
    private readonly MemoryMappedViewAccessor? _centroidsView;
    private readonly MemoryMappedFile? _offsetsMmf;
    private readonly MemoryMappedViewAccessor? _offsetsView;
    private readonly MemoryMappedFile? _bboxMinMmf;
    private readonly MemoryMappedViewAccessor? _bboxMinView;
    private readonly MemoryMappedFile? _bboxMaxMmf;
    private readonly MemoryMappedViewAccessor? _bboxMaxView;
    private readonly MemoryMappedFile? _pqCodebooksMmf;
    private readonly MemoryMappedViewAccessor? _pqCodebooksView;
    private readonly MemoryMappedFile? _pqCodesMmf;
    private readonly MemoryMappedViewAccessor? _pqCodesView;
    private readonly float* _vectorsPtr;
    private readonly byte* _labelsPtr;
    private readonly sbyte* _q8Ptr;
    private readonly sbyte* _q8SoaPtr;
    private readonly float* _centroidsPtr;
    private readonly int* _offsetsPtr;
    private readonly float* _bboxMinPtr;
    private readonly float* _bboxMaxPtr;
    private readonly float* _pqCodebooksPtr;
    private readonly byte* _pqCodesPtr;

    public int Count { get; }
    public int NumCells { get; }
    public int PqM { get; }
    public int PqKsub { get; }
    public bool HasQ8 => _q8Ptr != null;
    public bool HasQ8Soa => _q8SoaPtr != null;
    public bool HasIvf => _centroidsPtr != null && _offsetsPtr != null;
    public bool HasIvfBbox => _bboxMinPtr != null && _bboxMaxPtr != null;
    public bool HasPq => _pqCodebooksPtr != null && _pqCodesPtr != null;

    private Dataset(
        MemoryMappedFile vectorsMmf, MemoryMappedViewAccessor vectorsView,
        MemoryMappedFile labelsMmf, MemoryMappedViewAccessor labelsView,
        MemoryMappedFile? q8Mmf, MemoryMappedViewAccessor? q8View,
        MemoryMappedFile? q8SoaMmf, MemoryMappedViewAccessor? q8SoaView,
        MemoryMappedFile? centroidsMmf, MemoryMappedViewAccessor? centroidsView,
        MemoryMappedFile? offsetsMmf, MemoryMappedViewAccessor? offsetsView,
        MemoryMappedFile? bboxMinMmf, MemoryMappedViewAccessor? bboxMinView,
        MemoryMappedFile? bboxMaxMmf, MemoryMappedViewAccessor? bboxMaxView,
        MemoryMappedFile? pqCodebooksMmf, MemoryMappedViewAccessor? pqCodebooksView,
        MemoryMappedFile? pqCodesMmf, MemoryMappedViewAccessor? pqCodesView,
        int count, int numCells, int pqM, int pqKsub)
    {
        _vectorsMmf = vectorsMmf;
        _vectorsView = vectorsView;
        _labelsMmf = labelsMmf;
        _labelsView = labelsView;
        _q8Mmf = q8Mmf;
        _q8View = q8View;
        _q8SoaMmf = q8SoaMmf;
        _q8SoaView = q8SoaView;
        _centroidsMmf = centroidsMmf;
        _centroidsView = centroidsView;
        _offsetsMmf = offsetsMmf;
        _offsetsView = offsetsView;
        _bboxMinMmf = bboxMinMmf;
        _bboxMinView = bboxMinView;
        _bboxMaxMmf = bboxMaxMmf;
        _bboxMaxView = bboxMaxView;
        _pqCodebooksMmf = pqCodebooksMmf;
        _pqCodebooksView = pqCodebooksView;
        _pqCodesMmf = pqCodesMmf;
        _pqCodesView = pqCodesView;
        Count = count;
        NumCells = numCells;
        PqM = pqM;
        PqKsub = pqKsub;

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

        if (_pqCodebooksView is not null)
        {
            byte* cbBase = null;
            _pqCodebooksView.SafeMemoryMappedViewHandle.AcquirePointer(ref cbBase);
            _pqCodebooksPtr = (float*)cbBase;
        }

        if (_pqCodesView is not null)
        {
            byte* codesBase = null;
            _pqCodesView.SafeMemoryMappedViewHandle.AcquirePointer(ref codesBase);
            _pqCodesPtr = codesBase;
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
    public long Prefetch()
    {
        const int PageSize = 4096;
        long total = 0;
        long sink = 0;
        long hpTotal = 0;

        long vBytes = (long)Count * PaddedDimensions * sizeof(float);
        hpTotal += AdviseHuge((byte*)_vectorsPtr, vBytes);
        sink += TouchPages((byte*)_vectorsPtr, vBytes, PageSize);
        total += vBytes;

        long lBytes = Count;
        hpTotal += AdviseHuge(_labelsPtr, lBytes);
        sink += TouchPages(_labelsPtr, lBytes, PageSize);
        total += lBytes;

        if (_q8Ptr != null)
        {
            long bytes = (long)Count * 16; // q8 row stride = PaddedDimensions (16)
            hpTotal += AdviseHuge((byte*)_q8Ptr, bytes);
            sink += TouchPages((byte*)_q8Ptr, bytes, PageSize);
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
            total += bytes;
        }
        if (_offsetsPtr != null)
        {
            long bytes = (long)(NumCells + 1) * sizeof(int);
            hpTotal += AdviseHuge((byte*)_offsetsPtr, bytes);
            sink += TouchPages((byte*)_offsetsPtr, bytes, PageSize);
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
        if (_pqCodebooksPtr != null)
        {
            long bytes = (long)PqM * PqKsub * (Dimensions / PqM) * sizeof(float);
            hpTotal += AdviseHuge((byte*)_pqCodebooksPtr, bytes);
            sink += TouchPages((byte*)_pqCodebooksPtr, bytes, PageSize);
            total += bytes;
        }
        if (_pqCodesPtr != null)
        {
            long bytes = (long)Count * PqM;
            hpTotal += AdviseHuge(_pqCodesPtr, bytes);
            sink += TouchPages(_pqCodesPtr, bytes, PageSize);
            total += bytes;
        }

        GC.KeepAlive(sink);
        LastHugepageAdvisedBytes = hpTotal;
        return total;
    }

    /// <summary>Total bytes successfully advised with MADV_HUGEPAGE on the last Prefetch call. 0 on non-Linux or if disabled.</summary>
    public long LastHugepageAdvisedBytes { get; private set; }

    [DllImport("libc", EntryPoint = "madvise", SetLastError = true)]
    private static extern int LinuxMadvise(IntPtr addr, UIntPtr length, int advice);
    private const int MADV_HUGEPAGE = 14;
    private static readonly bool s_thpEnabled =
        Environment.GetEnvironmentVariable("DATASET_THP") != "0" &&
        OperatingSystem.IsLinux();

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
        string? ivfCentroidsPath = null,
        string? ivfOffsetsPath = null,
        string? ivfBboxMinPath = null,
        string? ivfBboxMaxPath = null,
        string? pqCodebooksPath = null,
        string? pqCodesPath = null,
        int pqM = 7,
        int pqKsub = 256)
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

        MemoryMappedFile? pqCodebooksMmf = null;
        MemoryMappedViewAccessor? pqCodebooksView = null;
        MemoryMappedFile? pqCodesMmf = null;
        MemoryMappedViewAccessor? pqCodesView = null;
        if (!string.IsNullOrEmpty(pqCodebooksPath) && File.Exists(pqCodebooksPath)
            && !string.IsNullOrEmpty(pqCodesPath) && File.Exists(pqCodesPath))
        {
            int dsub = Dimensions / pqM;
            long expectedCb = (long)pqM * pqKsub * dsub * sizeof(float);
            long cbLen = new FileInfo(pqCodebooksPath).Length;
            if (cbLen != expectedCb)
                throw new InvalidDataException($"PQ codebooks size {cbLen} != expected {expectedCb} (M={pqM} ksub={pqKsub} dsub={dsub})");
            long expectedCodes = (long)count * pqM;
            long codesLen = new FileInfo(pqCodesPath).Length;
            if (codesLen != expectedCodes)
                throw new InvalidDataException($"PQ codes size {codesLen} != expected {expectedCodes}");
            pqCodebooksMmf = MemoryMappedFile.CreateFromFile(pqCodebooksPath, FileMode.Open, null, 0, MemoryMappedFileAccess.Read);
            pqCodebooksView = pqCodebooksMmf.CreateViewAccessor(0, 0, MemoryMappedFileAccess.Read);
            pqCodesMmf = MemoryMappedFile.CreateFromFile(pqCodesPath, FileMode.Open, null, 0, MemoryMappedFileAccess.Read);
            pqCodesView = pqCodesMmf.CreateViewAccessor(0, 0, MemoryMappedFileAccess.Read);
        }

        return new Dataset(
            vectorsMmf, vectorsView, labelsMmf, labelsView,
            q8Mmf, q8View, q8SoaMmf, q8SoaView,
            centroidsMmf, centroidsView, offsetsMmf, offsetsView,
            bboxMinMmf, bboxMinView, bboxMaxMmf, bboxMaxView,
            pqCodebooksMmf, pqCodebooksView, pqCodesMmf, pqCodesView,
            (int)count, numCells, pqM, pqKsub);
    }

    public float* PqCodebooksPtr
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => _pqCodebooksPtr;
    }

    public byte* PqCodesPtr
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => _pqCodesPtr;
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
        try { _centroidsView?.SafeMemoryMappedViewHandle.ReleasePointer(); } catch { }
        try { _offsetsView?.SafeMemoryMappedViewHandle.ReleasePointer(); } catch { }
        try { _bboxMinView?.SafeMemoryMappedViewHandle.ReleasePointer(); } catch { }
        try { _bboxMaxView?.SafeMemoryMappedViewHandle.ReleasePointer(); } catch { }
        try { _pqCodebooksView?.SafeMemoryMappedViewHandle.ReleasePointer(); } catch { }
        try { _pqCodesView?.SafeMemoryMappedViewHandle.ReleasePointer(); } catch { }
        _vectorsView.Dispose();
        _vectorsMmf.Dispose();
        _labelsView.Dispose();
        _labelsMmf.Dispose();
        _q8View?.Dispose();
        _q8Mmf?.Dispose();
        _q8SoaView?.Dispose();
        _q8SoaMmf?.Dispose();
        _centroidsView?.Dispose();
        _centroidsMmf?.Dispose();
        _offsetsView?.Dispose();
        _offsetsMmf?.Dispose();
        _bboxMinView?.Dispose();
        _bboxMinMmf?.Dispose();
        _bboxMaxView?.Dispose();
        _bboxMaxMmf?.Dispose();
        _pqCodebooksView?.Dispose();
        _pqCodebooksMmf?.Dispose();
        _pqCodesView?.Dispose();
        _pqCodesMmf?.Dispose();
    }
}
