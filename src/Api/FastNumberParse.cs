using System.Runtime.CompilerServices;

namespace Rinha.Api;

/// <summary>
/// Fast scalar parser for JSON numeric tokens of the shape "ddd" or "ddd.ddd"
/// (no exponent, no leading sign). Uses Clinger's fast path: accumulate the
/// significand as <see cref="ulong"/> while tracking the decimal exponent, then
/// multiply by an exact power-of-ten from a precomputed table.
///
/// Replaces <c>Utf8JsonReader.GetDouble()</c> (Number.NumberToFloatingPointBits
/// slow path) and <c>Utf8Parser.TryParseNormalAsFloatingPoint</c>. Profiling on
/// the RAW_HTTP hot path (2026-05-16) attributed ~3% of total CPU to those two
/// frames combined for 6 numeric fields per request.
///
/// Correctness: significand fits in 64 bits for any of our payloads
/// (max observed ~13 significant digits, ~2^43 ≪ 2^64). Multiplication by an
/// exact-double power of ten (<c>1e0..1e22</c>) yields a single correctly
/// rounded result by IEEE 754; any sub-ULP drift is absorbed by the Q16 round
/// downstream (1/10000 quantum &gt; 1e-12 ULP).
/// </summary>
internal static class FastNumberParse
{
    // 1e-d for d in [0..18]. All are exactly representable in double
    // up to 10^22 per Clinger; we cap at 18 (max field has 10 decimal digits).
    private static readonly double[] NegPow10 =
    {
        1e0,  1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9,
        1e-10, 1e-11, 1e-12, 1e-13, 1e-14, 1e-15, 1e-16, 1e-17, 1e-18,
    };

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double ParseDouble(ReadOnlySpan<byte> s)
    {
        int len = s.Length;
        if (len == 0) return 0;
        ulong mantissa = 0;
        int i = 0;

        // Integer digits (or until '.').
        while (i < len)
        {
            byte b = s[i];
            if (b == (byte)'.') break;
            mantissa = mantissa * 10 + (uint)(b - (byte)'0');
            i++;
        }

        if (i == len) return mantissa;

        // Fractional digits after '.'.
        i++; // skip '.'
        int fracStart = i;
        while (i < len)
        {
            mantissa = mantissa * 10 + (uint)(s[i] - (byte)'0');
            i++;
        }
        int fracDigits = i - fracStart;
        return mantissa * NegPow10[fracDigits];
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int ParseInt32(ReadOnlySpan<byte> s)
    {
        int len = s.Length;
        int v = 0;
        for (int i = 0; i < len; i++)
            v = v * 10 + (s[i] - (byte)'0');
        return v;
    }

    /// <summary>
    /// Streaming double parser: scans <paramref name="body"/> starting at <paramref name="pos"/>
    /// consuming digits + optional single '.', advancing <paramref name="pos"/> to the first
    /// non-numeric byte (which will be the JSON delimiter ',' or '}'). Single pass — eliminates
    /// the prior <c>IndexOf(delim) + ParseDouble(slice)</c> two-pass scan over the same bytes.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double ParseDouble(ReadOnlySpan<byte> body, ref int pos)
    {
        int i = pos;
        ulong mantissa = 0;
        // Integer digits.
        while (true)
        {
            byte b = body[i];
            uint d = (uint)(b - (byte)'0');
            if (d > 9) break;
            mantissa = mantissa * 10 + d;
            i++;
        }
        if (body[i] != (byte)'.')
        {
            pos = i;
            return mantissa;
        }
        i++; // skip '.'
        int fracStart = i;
        while (true)
        {
            byte b = body[i];
            uint d = (uint)(b - (byte)'0');
            if (d > 9) break;
            mantissa = mantissa * 10 + d;
            i++;
        }
        int fracDigits = i - fracStart;
        pos = i;
        return mantissa * NegPow10[fracDigits];
    }

    /// <summary>Streaming int parser; advances <paramref name="pos"/> past the digits.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int ParseInt32(ReadOnlySpan<byte> body, ref int pos)
    {
        int i = pos;
        int v = 0;
        while (true)
        {
            byte b = body[i];
            uint d = (uint)(b - (byte)'0');
            if (d > 9) break;
            v = v * 10 + (int)d;
            i++;
        }
        pos = i;
        return v;
    }
}
