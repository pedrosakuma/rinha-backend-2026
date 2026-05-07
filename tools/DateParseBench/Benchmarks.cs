using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;

/// <summary>
/// Compares three implementations of parsing the canonical ISO 8601 UTC timestamp
/// "YYYY-MM-DDTHH:MM:SSZ" (exactly 20 ASCII bytes) into a UTC ticks value.
///
/// Scalar    — mirrors src/Api/JsonVectorizer.cs ParseIsoUtc: D2/D4 byte reads + new DateTime(...)
/// UlongSwar — two Unsafe.ReadUnaligned&lt;ulong&gt; reads, then shift/mask to extract fields
/// Vector128Swar — one 16-byte SIMD load + pshufb gather + pmaddubsw digit-pair combine (SSSE3)
/// </summary>
[MemoryDiagnoser]
[ShortRunJob]
public class DateParseBenchmarks
{
    // Precomputed static masks — hoisted out of the hot path by the JIT.
    // Gather digit bytes to contiguous positions: [y3,y2,y1,y0, m1,m0, d1,d0, h1,h0, mi1,mi0, pad...]
    private static readonly Vector128<byte> s_shuffleMask = Vector128.Create(
        (byte)0, 1, 2, 3,   // year YYYY  (positions 0-3 in the 20-byte string)
        5, 6,               // month MM   (positions 5-6)
        8, 9,               // day DD     (positions 8-9)
        11, 12,             // hour HH    (positions 11-12)
        14, 15,             // minute MM  (positions 14-15)
        0, 0, 0, 0          // padding (unused; replicate byte 0 to keep mask constant)
    );

    // pmaddubsw coefficients: alternating *10 / *1 combines each digit pair into one int16.
    // result[i] = gathered[2i]*10 + gathered[2i+1]
    private static readonly Vector128<sbyte> s_mulMask = Vector128.Create(
        (sbyte)10, 1, 10, 1,   // year high pair (th*10+hu), year low pair (te*10+un)
        10, 1,                  // month
        10, 1,                  // day
        10, 1,                  // hour
        10, 1,                  // minute
        0, 0, 0, 0             // padding
    );

    private byte[] _input = null!;

    [GlobalSetup]
    public void Setup()
    {
        _input = "2026-05-07T14:23:45Z"u8.ToArray();

        long expected = new DateTime(2026, 5, 7, 14, 23, 45, DateTimeKind.Utc).Ticks;
        long ticksScalar = ParseScalar(_input);
        long ticksUlong  = ParseUlongSwar(_input);
        long ticksVec    = ParseVector128Swar(_input);

        if (ticksScalar != expected)
            throw new Exception($"Scalar mismatch: got {ticksScalar}, expected {expected}");
        if (ticksUlong != expected)
            throw new Exception($"UlongSwar mismatch: got {ticksUlong}, expected {expected}");
        if (ticksVec != expected)
            throw new Exception($"Vector128Swar mismatch: got {ticksVec}, expected {expected}");
    }

    // -------------------------------------------------------------------------
    // Benchmark entry points
    // -------------------------------------------------------------------------

    [Benchmark(Baseline = true)]
    public long Scalar() => ParseScalar(_input);

    [Benchmark]
    public long UlongSwar() => ParseUlongSwar(_input);

    [Benchmark]
    public long Vector128Swar() => ParseVector128Swar(_input);

    // =========================================================================
    // Implementation: Scalar
    // Mirrors src/Api/JsonVectorizer.cs ParseIsoUtc exactly.
    // =========================================================================

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int D2(ReadOnlySpan<byte> s, int o)
        => (s[o] - '0') * 10 + (s[o + 1] - '0');

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int D4(ReadOnlySpan<byte> s, int o)
        => (s[o] - '0') * 1000 + (s[o + 1] - '0') * 100
         + (s[o + 2] - '0') * 10 + (s[o + 3] - '0');

    private static long ParseScalar(byte[] input)
    {
        ReadOnlySpan<byte> s = input;
        int y  = D4(s, 0);
        int mo = D2(s, 5);
        int d  = D2(s, 8);
        int h  = D2(s, 11);
        int mi = D2(s, 14);
        int se = D2(s, 17);
        return new DateTime(y, mo, d, h, mi, se, DateTimeKind.Utc).Ticks;
    }

    // =========================================================================
    // Implementation: UlongSwar
    //
    // Reads 16 bytes as 2 × ulong (Unsafe.ReadUnaligned — no alignment
    // requirement). Digits are isolated with a nibble AND (& 0x0F per byte)
    // instead of subtracting '0' from the whole word: subtraction of
    // 0x3030…30 propagates borrows across byte boundaries when any byte
    // is a separator ('-', 'T', ':') whose ASCII value < '0'. The nibble
    // AND (& 0x0F0F…0F) has no inter-byte carry, and for ASCII digits
    // '0'–'9' the lower nibble equals the decimal digit value.
    //
    // Memory layout (little-endian):
    //   u0 byte[0..7]  = "2026-05-"  → y3 y2 y1 y0 sep m1 m0 sep
    //   u1 byte[8..15] = "07T14:23"  → d1 d0 sep h1 h0 sep mi1 mi0
    //   tail[16..19]   = ":45Z"      → sep s1 s0 Z   (scalar)
    // =========================================================================

    private static long ParseUlongSwar(byte[] input)
    {
        ref byte r = ref MemoryMarshal.GetReference((ReadOnlySpan<byte>)input);

        ulong u0 = Unsafe.ReadUnaligned<ulong>(ref r);
        ulong u1 = Unsafe.ReadUnaligned<ulong>(ref Unsafe.Add(ref r, 8));

        // Isolate the decimal digit from each ASCII byte via nibble mask.
        // Separator bytes get wrong nibble values, but those positions are
        // never extracted below, so correctness is preserved.
        const ulong nibble = 0x0F0F0F0F0F0F0F0FUL;
        u0 &= nibble;
        u1 &= nibble;

        // Year: bytes 0,1,2,3 of u0 (little-endian → shifts 0, 8, 16, 24)
        int y = (int)( u0        & 0xFF) * 1000
              + (int)((u0 >>  8) & 0xFF) * 100
              + (int)((u0 >> 16) & 0xFF) * 10
              + (int)((u0 >> 24) & 0xFF);

        // Month: bytes 5,6 of u0 (shifts 40, 48); bytes 4 and 7 are '-' — ignored
        int mo = (int)((u0 >> 40) & 0xFF) * 10
               + (int)((u0 >> 48) & 0xFF);

        // Day: bytes 0,1 of u1 (shifts 0, 8); byte 2 is 'T' — ignored
        int d = (int)(u1         & 0xFF) * 10
              + (int)((u1 >>  8) & 0xFF);

        // Hour: bytes 3,4 of u1 (shifts 24, 32); byte 5 is ':' — ignored
        int h = (int)((u1 >> 24) & 0xFF) * 10
              + (int)((u1 >> 32) & 0xFF);

        // Minute: bytes 6,7 of u1 (shifts 48, 56)
        int mi = (int)((u1 >> 48) & 0xFF) * 10
               + (int)((u1 >> 56) & 0xFF);

        // Tail bytes 17,18 are the second digits of seconds (":45Z")
        int se = (Unsafe.Add(ref r, 17) - (byte)'0') * 10
               + (Unsafe.Add(ref r, 18) - (byte)'0');

        return CivilToTicks(y, mo, d, h, mi, se);
    }

    // =========================================================================
    // Implementation: Vector128Swar
    //
    // Loads 16 bytes in a single SIMD instruction (covers "YYYY-MM-DDTHH:MM").
    // Uses pshufb (Ssse3.Shuffle) to gather the 12 digit bytes to contiguous
    // positions, then pmaddubsw (Ssse3.MultiplyAddAdjacent) to combine digit
    // pairs (tens×10 + units) into int16 results in one instruction.
    // Seconds tail (bytes 17-18) is handled scalar.
    // Falls back to element-wise extraction if SSSE3 is absent.
    // =========================================================================

    private static long ParseVector128Swar(byte[] input)
    {
        ref byte r = ref MemoryMarshal.GetReference((ReadOnlySpan<byte>)input);

        // Load the first 16 bytes: "YYYY-MM-DDTHH:MM"
        var v = Vector128.LoadUnsafe(ref r);
        // Subtract '0' from every byte. Separator positions get wrap-around
        // values that are discarded by the subsequent shuffle.
        var v2 = v - Vector128.Create((byte)'0');

        int y, mo, d, h, mi;

        if (Ssse3.IsSupported)
        {
            // pshufb: gather the 12 digit bytes into positions 0..11
            var gathered = Ssse3.Shuffle(v2, s_shuffleMask);

            // pmaddubsw: for each adjacent pair (a, b) → a*10 + b → int16
            // result[0] = y_hi = y_thousands*10 + y_hundreds  (e.g. 20)
            // result[1] = y_lo = y_tens*10      + y_units      (e.g. 26)
            // result[2..5] = month, day, hour, minute (single-step)
            Vector128<short> pairs = Ssse3.MultiplyAddAdjacent(gathered, s_mulMask);

            y  = pairs[0] * 100 + pairs[1];
            mo = pairs[2];
            d  = pairs[3];
            h  = pairs[4];
            mi = pairs[5];
        }
        else
        {
            // Fallback path: element extraction from the preloaded vector.
            y  = v2[0] * 1000 + v2[1] * 100 + v2[2] * 10 + v2[3];
            mo = v2[5] * 10 + v2[6];
            d  = v2[8] * 10 + v2[9];
            h  = v2[11] * 10 + v2[12];
            mi = v2[14] * 10 + v2[15];
        }

        int se = (Unsafe.Add(ref r, 17) - (byte)'0') * 10
               + (Unsafe.Add(ref r, 18) - (byte)'0');

        return CivilToTicks(y, mo, d, h, mi, se);
    }

    // =========================================================================
    // Date arithmetic: Howard Hinnant's civil-to-days algorithm
    // https://howardhinnant.github.io/date_algorithms.html#civil_from_days
    //
    // Converts (y, m, d, h, mi, s) → .NET ticks without allocating DateTime.
    // .NET epoch offset: DateTime.UnixEpoch.Ticks = 621_355_968_000_000_000L
    //   = 719162 days × 86400 s/day × 10_000_000 ticks/s
    // =========================================================================

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static long CivilToTicks(int y, int m, int d, int h, int mi, int s)
    {
        // Shift the calendar so the year starts on March 1 (puts the leap day
        // at the end of the shifted year, simplifying leap-year arithmetic).
        int yp  = m <= 2 ? y - 1 : y;
        int era = (yp >= 0 ? yp : yp - 399) / 400;   // 400-year era
        int yoe = yp - era * 400;                      // year-of-era [0, 399]
        int doy = (153 * (m + (m > 2 ? -3 : 9)) + 2) / 5 + d - 1; // day-of-shifted-year [0, 365]
        int doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;            // day-of-era [0, 146096]

        // Days since Unix epoch (1970-01-01). Magic constant 719468 = days
        // from the proleptic-Gregorian epoch to 1970-03-01 in the shifted calendar.
        long unixDays = (long)era * 146097L + doe - 719468L;

        return (unixDays * 86400L + h * 3600L + mi * 60L + s) * 10_000_000L
             + 621_355_968_000_000_000L;
    }
}
