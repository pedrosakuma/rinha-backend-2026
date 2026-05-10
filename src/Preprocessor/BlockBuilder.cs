using System.Buffers.Binary;
using System.Diagnostics;

namespace Rinha.Preprocessor;

/// <summary>
/// Builds the Block-SoA Q16 reference dataset used by IvfBlockedScorer.
///
/// Inputs:
///   references_q16.bin       (N × 16 i16, AoS post-IVF reorder; 14 dims used + 2 padded)
///   ivf_offsets.bin          ((NLIST+1) × i32, prefix sum of cell sizes; cell c → [off[c], off[c+1]))
///
/// Outputs:
///   references_q16_blocked.bin (TotalBlocks × 112 i16 = TotalBlocks × 224 bytes).
///                               Layout per block: [d0_lane0..d0_lane7][d1_lane0..d1_lane7]...[d13_lane0..d13_lane7].
///                               Padding lanes (last block of cell when size % 8 != 0) filled with short.MaxValue.
///   ivf_block_offsets.bin       ((NLIST+1) × i32, prefix sum of blocks per cell).
///
/// Block layout matches jairoblatt/rinha-2026-rust src/build_index.rs (8 lanes × 14 dims, dim-major within block).
/// MaxValue padding ensures invalid lanes never beat the running top-K threshold (their distance² overflows but
/// is filtered by the partial early-abort check; in practice we also use the labels offset to skip out-of-cell rows).
/// </summary>
public static class BlockBuilder
{
    private const int Dimensions = 14;
    private const int PaddedDimensions = 16;
    private const int BlockSize = 8;
    private const int BlockShorts = BlockSize * Dimensions; // 112 i16/block

    public static int Run(string[] args)
    {
        if (args.Length < 4)
        {
            Console.Error.WriteLine(
                "Usage: Rinha.Preprocessor --block <vectors_q16.bin> <ivf_offsets.bin> " +
                "<out_blocked.bin> <out_block_offsets.bin>");
            return 1;
        }

        var q16Path = args[0];
        var offsetsPath = args[1];
        var blockedPath = args[2];
        var blockOffsetsPath = args[3];

        var sw = Stopwatch.StartNew();

        var q16Len = new FileInfo(q16Path).Length;
        long rowBytes = PaddedDimensions * sizeof(short);
        if (q16Len % rowBytes != 0)
            throw new InvalidDataException($"Q16 size {q16Len} not multiple of {rowBytes}");
        long count = q16Len / rowBytes;
        if (count > int.MaxValue)
            throw new InvalidDataException("Dataset too large");
        int n = (int)count;

        var offsetsLen = new FileInfo(offsetsPath).Length;
        if (offsetsLen % sizeof(int) != 0)
            throw new InvalidDataException($"Offsets size {offsetsLen} not multiple of 4");
        int numCellsPlus1 = (int)(offsetsLen / sizeof(int));
        if (numCellsPlus1 < 2)
            throw new InvalidDataException("Offsets file must have at least 2 ints");
        int numCells = numCellsPlus1 - 1;

        var offsetsBytes = File.ReadAllBytes(offsetsPath);
        var offsets = new int[numCellsPlus1];
        Buffer.BlockCopy(offsetsBytes, 0, offsets, 0, offsetsBytes.Length);
        if (offsets[numCells] != n)
            throw new InvalidDataException($"Offsets last value {offsets[numCells]} != count {n}");

        // Compute total blocks (ceil(cell_size / 8) per cell).
        int totalBlocks = 0;
        var blockOffsets = new int[numCellsPlus1];
        for (int c = 0; c < numCells; c++)
        {
            blockOffsets[c] = totalBlocks;
            int sz = offsets[c + 1] - offsets[c];
            int blocks = (sz + BlockSize - 1) / BlockSize;
            totalBlocks += blocks;
        }
        blockOffsets[numCells] = totalBlocks;

        Console.Error.WriteLine($"BlockBuilder: N={n:N0} NLIST={numCells} totalBlocks={totalBlocks:N0}");
        Console.Error.WriteLine($"  blocked size = {(long)totalBlocks * BlockShorts * sizeof(short):N0} bytes");

        // Read full Q16 (rowBytes per row).
        var q16 = new short[(long)n * PaddedDimensions];
        ReadAllShorts(q16Path, q16);

        // Allocate output: totalBlocks * 112 shorts. Initialize all to short.MaxValue so
        // padding lanes don't need explicit fill in the inner loop.
        var blocked = new short[(long)totalBlocks * BlockShorts];
        // Initialize buffer to short.MaxValue so that any lane we don't write (padding when
        // cell size % 8 != 0) yields a huge (q - MaxValue)² that loses every top-K race.
        Array.Fill(blocked, short.MaxValue);

        // Pack each cell's rows into block-SoA layout: for each block of up to 8 rows,
        // for each dim d in [0..14), write the 8 lanes contiguously.
        int blockIdx = 0;
        for (int c = 0; c < numCells; c++)
        {
            int rowStart = offsets[c];
            int rowEnd = offsets[c + 1];
            int rowsInCell = rowEnd - rowStart;
            int blocksInCell = (rowsInCell + BlockSize - 1) / BlockSize;
            for (int b = 0; b < blocksInCell; b++)
            {
                int laneRowBase = rowStart + b * BlockSize;
                int lanesUsed = Math.Min(BlockSize, rowEnd - laneRowBase);
                long dst = (long)(blockIdx + b) * BlockShorts;
                for (int d = 0; d < Dimensions; d++)
                {
                    long dstDim = dst + (long)d * BlockSize;
                    for (int lane = 0; lane < lanesUsed; lane++)
                    {
                        int row = laneRowBase + lane;
                        blocked[dstDim + lane] = q16[(long)row * PaddedDimensions + d];
                    }
                    // lanes [lanesUsed..8) already short.MaxValue from Array.Fill above.
                }
            }
            blockIdx += blocksInCell;
        }

        // Write outputs.
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(blockedPath))!);
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(blockOffsetsPath))!);

        WriteAllShorts(blockedPath, blocked);

        var boBytes = new byte[blockOffsets.Length * sizeof(int)];
        Buffer.BlockCopy(blockOffsets, 0, boBytes, 0, boBytes.Length);
        File.WriteAllBytes(blockOffsetsPath, boBytes);

        sw.Stop();
        Console.Error.WriteLine($"BlockBuilder done in {sw.Elapsed}.");
        Console.Error.WriteLine($"Blocked:        {blockedPath} ({new FileInfo(blockedPath).Length:N0} bytes)");
        Console.Error.WriteLine($"Block-offsets:  {blockOffsetsPath} ({new FileInfo(blockOffsetsPath).Length:N0} bytes)");
        return 0;
    }

    private static void ReadAllShorts(string path, short[] dst)
    {
        using var fs = File.OpenRead(path);
        var bytes = new byte[dst.Length * sizeof(short)];
        int read = 0;
        while (read < bytes.Length)
        {
            int got = fs.Read(bytes, read, bytes.Length - read);
            if (got == 0) throw new EndOfStreamException(path);
            read += got;
        }
        Buffer.BlockCopy(bytes, 0, dst, 0, bytes.Length);
    }

    private static void WriteAllShorts(string path, short[] src)
    {
        var bytes = new byte[(long)src.Length * sizeof(short)];
        Buffer.BlockCopy(src, 0, bytes, 0, bytes.Length);
        File.WriteAllBytes(path, bytes);
    }
}
