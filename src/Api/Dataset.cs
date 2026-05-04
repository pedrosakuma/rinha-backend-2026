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
    private readonly MemoryMappedFile? _centroidsMmf;
    private readonly MemoryMappedViewAccessor? _centroidsView;
    private readonly MemoryMappedFile? _offsetsMmf;
    private readonly MemoryMappedViewAccessor? _offsetsView;
    private readonly float* _vectorsPtr;
    private readonly byte* _labelsPtr;
    private readonly sbyte* _q8Ptr;
    private readonly float* _centroidsPtr;
    private readonly int* _offsetsPtr;

    public int Count { get; }
    public int NumCells { get; }
    public bool HasQ8 => _q8Ptr != null;
    public bool HasIvf => _centroidsPtr != null && _offsetsPtr != null;

    private Dataset(
        MemoryMappedFile vectorsMmf, MemoryMappedViewAccessor vectorsView,
        MemoryMappedFile labelsMmf, MemoryMappedViewAccessor labelsView,
        MemoryMappedFile? q8Mmf, MemoryMappedViewAccessor? q8View,
        MemoryMappedFile? centroidsMmf, MemoryMappedViewAccessor? centroidsView,
        MemoryMappedFile? offsetsMmf, MemoryMappedViewAccessor? offsetsView,
        int count, int numCells)
    {
        _vectorsMmf = vectorsMmf;
        _vectorsView = vectorsView;
        _labelsMmf = labelsMmf;
        _labelsView = labelsView;
        _q8Mmf = q8Mmf;
        _q8View = q8View;
        _centroidsMmf = centroidsMmf;
        _centroidsView = centroidsView;
        _offsetsMmf = offsetsMmf;
        _offsetsView = offsetsView;
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
    }

    public static Dataset Open(
        string vectorsPath,
        string labelsPath,
        string? vectorsQ8Path = null,
        string? ivfCentroidsPath = null,
        string? ivfOffsetsPath = null)
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

        return new Dataset(
            vectorsMmf, vectorsView, labelsMmf, labelsView,
            q8Mmf, q8View, centroidsMmf, centroidsView, offsetsMmf, offsetsView,
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

    public void Dispose()
    {
        try { _vectorsView.SafeMemoryMappedViewHandle.ReleasePointer(); } catch { }
        try { _labelsView.SafeMemoryMappedViewHandle.ReleasePointer(); } catch { }
        try { _q8View?.SafeMemoryMappedViewHandle.ReleasePointer(); } catch { }
        try { _centroidsView?.SafeMemoryMappedViewHandle.ReleasePointer(); } catch { }
        try { _offsetsView?.SafeMemoryMappedViewHandle.ReleasePointer(); } catch { }
        _vectorsView.Dispose();
        _vectorsMmf.Dispose();
        _labelsView.Dispose();
        _labelsMmf.Dispose();
        _q8View?.Dispose();
        _q8Mmf?.Dispose();
        _centroidsView?.Dispose();
        _centroidsMmf?.Dispose();
        _offsetsView?.Dispose();
        _offsetsMmf?.Dispose();
    }
}
