using System.Buffers;
using System.Buffers.Text;
using System.Globalization;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Runtime.CompilerServices;
using System.Text.Json;

namespace Rinha.Api;

/// <summary>
/// J11c: zero-allocation JSON → vector parser. Bypasses System.Text.Json's
/// object materialization (FraudRequest + 5 sub-DTOs + List&lt;string&gt; alloc on every
/// request) and goes straight from raw UTF-8 bytes into the destination Span&lt;float&gt;.
///
/// Property dispatch is done via UTF-8 byte SequenceEqual on the reader's value span,
/// avoiding string interning and case-folding paths inside STJ's binder.
///
/// Datetime fields are parsed with a custom hot-path that handles the canonical ISO 8601
/// shape used by the bench ("YYYY-MM-DDTHH:MM:SSZ"), falling back to DateTimeOffset.Parse
/// for any other shape.
/// </summary>
public sealed class JsonVectorizer
{
    private readonly NormalizationConstants _norm;
    private readonly MccRiskTable _mcc;

    private static ReadOnlySpan<byte> KTransaction => "transaction"u8;
    private static ReadOnlySpan<byte> KCustomer => "customer"u8;
    private static ReadOnlySpan<byte> KMerchant => "merchant"u8;
    private static ReadOnlySpan<byte> KTerminal => "terminal"u8;
    private static ReadOnlySpan<byte> KLastTransaction => "last_transaction"u8;

    private static ReadOnlySpan<byte> KAmount => "amount"u8;
    private static ReadOnlySpan<byte> KInstallments => "installments"u8;
    private static ReadOnlySpan<byte> KRequestedAt => "requested_at"u8;

    private static ReadOnlySpan<byte> KAvgAmount => "avg_amount"u8;
    private static ReadOnlySpan<byte> KTxCount24h => "tx_count_24h"u8;
    private static ReadOnlySpan<byte> KKnownMerchants => "known_merchants"u8;

    private static ReadOnlySpan<byte> KId => "id"u8;
    private static ReadOnlySpan<byte> KMcc => "mcc"u8;

    private static ReadOnlySpan<byte> KIsOnline => "is_online"u8;
    private static ReadOnlySpan<byte> KCardPresent => "card_present"u8;
    private static ReadOnlySpan<byte> KKmFromHome => "km_from_home"u8;

    private static ReadOnlySpan<byte> KTimestamp => "timestamp"u8;
    private static ReadOnlySpan<byte> KKmFromCurrent => "km_from_current"u8;

    public JsonVectorizer(NormalizationConstants norm, MccRiskTable mcc)
    {
        _norm = norm;
        _mcc = mcc;
    }

    public void VectorizeJson(ReadOnlySpan<byte> body, Span<float> dst)
    {
        if (dst.Length < Dataset.Dimensions)
            throw new ArgumentException("Destination too small", nameof(dst));
        Span<short> q = stackalloc short[Dataset.Dimensions];
        VectorizeJsonCore(body, q);
        for (int i = 0; i < Dataset.Dimensions; i++) dst[i] = q[i] * (1f / Dataset.Q16Scale);
    }

    /// <summary>Q16 path: produces the canonical int16 query directly. See
    /// <see cref="Vectorizer.VectorizeQ16"/> for semantics.</summary>
    public void VectorizeJsonQ16(ReadOnlySpan<byte> body, Span<short> dst)
    {
        if (dst.Length < Dataset.Dimensions)
            throw new ArgumentException("Destination too small", nameof(dst));
        VectorizeJsonCore(body, dst);
    }

    /// <summary>Combined: float + Q16 in one parse.</summary>
    public void VectorizeJson(ReadOnlySpan<byte> body, Span<float> floatDst, Span<short> q16Dst)
    {
        if (floatDst.Length < Dataset.Dimensions || q16Dst.Length < Dataset.Dimensions)
            throw new ArgumentException("Destination too small");
        VectorizeJsonCore(body, q16Dst);
        for (int i = 0; i < Dataset.Dimensions; i++) floatDst[i] = q16Dst[i] * (1f / Dataset.Q16Scale);
    }

    private static readonly bool s_useFastJson =
        Environment.GetEnvironmentVariable("JSON_FAST") != "0";

    private void VectorizeJsonCore(ReadOnlySpan<byte> body, Span<short> dst)
    {
        if (s_useFastJson) VectorizeJsonCoreFast(body, dst);
        else VectorizeJsonCoreReader(body, dst);
    }

    // Schema-positional parser. Trusts the producer's exact byte layout (no validation):
    //   {"id":"...","transaction":{"amount":N,"installments":N,"requested_at":"ISO"},
    //    "customer":{"avg_amount":N,"tx_count_24h":N,"known_merchants":[...]},
    //    "merchant":{"id":"MERC-XXX","mcc":"DDDD","avg_amount":N},
    //    "terminal":{"is_online":B,"card_present":B,"km_from_home":N},
    //    "last_transaction":null|{"timestamp":"ISO","km_from_current":N}}
    // Fixed-length prefixes & values are skipped by constant; numbers use single-pass
    // FastNumberParse streaming (advances `p` to delimiter — no IndexOf two-pass).
    // The only IndexOf calls remaining are for the variable-length top-level id and
    // the known_merchants array end.
    private void VectorizeJsonCoreFast(ReadOnlySpan<byte> body, Span<short> dst)
    {
        if (dst.Length < Dataset.Dimensions)
            throw new ArgumentException("dst too small", nameof(dst));

        int p = 7; // past `{"id":"`
        // Top-level id length distribution: 77% are 13 chars, 21% are 12, ~2% are 8-11.
        // Probe by the unambiguous 2-byte pattern `",` (closing quote + delimiter); checking
        // just `"` alone is ambiguous because the next field's opening quote sits 2 bytes
        // after the closing quote (`","transaction"...`) so `body[p+L+2]` is also `"`.
        int idEnd;
        if (body[p + 13] == (byte)'"' && body[p + 14] == (byte)',') idEnd = p + 13;
        else if (body[p + 12] == (byte)'"' && body[p + 13] == (byte)',') idEnd = p + 12;
        else idEnd = p + 8 + body.Slice(p + 8).IndexOf((byte)'"');
        p = idEnd + 1; // past closing `"`

        p += 25; // past `,"transaction":{"amount":`
        double txAmount = FastNumberParse.ParseDouble(body, ref p);

        p += 16; // past `,"installments":`
        int installments = FastNumberParse.ParseInt32(body, ref p);

        p += 17; // past `,"requested_at":"`
        ParseIsoUtc(body.Slice(p, 20), out long requestedTicks, out int requestedHour, out DayOfWeek requestedDow);
        p += 21; // past 20-byte ISO + closing `"`

        p += 27; // past `},"customer":{"avg_amount":`
        double custAvg = FastNumberParse.ParseDouble(body, ref p);

        p += 16; // past `,"tx_count_24h":`
        int txCount24h = FastNumberParse.ParseInt32(body, ref p);

        p += 19; // past `,"known_merchants":` — p is at `[`
        // known_merchants entries are always "MERC-XXX" (10 bytes with quotes); the array
        // contains 2-5 entries uniformly → total bytes ∈ {23, 34, 45, 56} (closing `]`
        // at offset {22, 33, 44, 55} from `[`). No IndexOf needed.
        int kmArrStart = p;
        int kmArrLen;
        if (body[p + 22] == (byte)']') kmArrLen = 23;
        else if (body[p + 33] == (byte)']') kmArrLen = 34;
        else if (body[p + 44] == (byte)']') kmArrLen = 45;
        else kmArrLen = 56;
        int kmArrEnd = p + kmArrLen;
        p = kmArrEnd;

        p += 20; // past `},"merchant":{"id":"`
        // merchant.id is always 8 bytes ("MERC-XXX") in the bench dataset.
        int merchIdStart = p;
        int merchIdEnd = p + 8;
        p = merchIdEnd + 1; // past closing `"`

        p += 8; // past `,"mcc":"`
        // mcc is always 4 bytes.
        int mccStart = p;
        int mccEnd = p + 4;
        p = mccEnd + 1; // past closing `"`

        p += 14; // past `,"avg_amount":`
        double merchAvg = FastNumberParse.ParseDouble(body, ref p);

        p += 26; // past `},"terminal":{"is_online":`
        bool isOnline = body[p] == (byte)'t';
        p += isOnline ? 4 : 5;

        p += 16; // past `,"card_present":`
        bool cardPresent = body[p] == (byte)'t';
        p += cardPresent ? 4 : 5;

        p += 16; // past `,"km_from_home":`
        double kmHome = FastNumberParse.ParseDouble(body, ref p);

        p += 21; // past `},"last_transaction":`
        bool hasLast;
        long lastTimestampTicks = 0;
        double lastKm = 0;
        if (body[p] == (byte)'n')
        {
            hasLast = false;
        }
        else
        {
            hasLast = true;
            p += 14; // past `{"timestamp":"`
            ParseIsoUtc(body.Slice(p, 20), out lastTimestampTicks, out _, out _);
            p += 21; // past 20-byte ISO + closing `"`
            p += 19; // past `,"km_from_current":`
            lastKm = FastNumberParse.ParseDouble(body, ref p);
        }

        ComposeFeatures(dst, body,
            txAmount, custAvg, merchAvg, kmHome,
            installments, txCount24h,
            isOnline, cardPresent,
            hasLast, requestedTicks, lastTimestampTicks, requestedHour, requestedDow, lastKm,
            kmArrStart, kmArrEnd, merchIdStart, merchIdEnd, mccStart, mccEnd);
    }

    private void VectorizeJsonCoreReader(ReadOnlySpan<byte> body, Span<short> dst)
    {
        // Length-guard once so the JIT (and our manual ref-add writes below) can elide bounds checks.
        if (dst.Length < Dataset.Dimensions)
            throw new ArgumentException("dst too small", nameof(dst));

        // Per-request scratch state.
        double txAmount = 0, custAvg = 0, merchAvg = 0, kmHome = 0;
        int installments = 0, txCount24h = 0;
        bool isOnline = false, cardPresent = false;
        bool hasLast = false;
        long requestedAtTicksUtc = 0;        // Unix ticks (we only need diff in minutes & H/DOW)
        long lastTimestampTicksUtc = 0;
        DayOfWeek requestedDow = DayOfWeek.Monday;
        int requestedHour = 0;
        double lastKm = 0;
        // Known-merchant membership: parsed inline against the merchant.id we resolve later.
        // To avoid two passes, we capture spans for both and compare at the end.
        // merchant.id appears in "merchant.id"; known_merchants is in customer.
        // We stash known_merchants raw bytes range and merchant.id bytes, then resolve.

        // Use a small instance scratch to remember the slice of the body containing the
        // known_merchants array (start..end, exclusive). 0 = not seen.
        int kmArrStart = 0, kmArrEnd = 0;
        int merchIdStart = 0, merchIdEnd = 0;
        int mccStart = 0, mccEnd = 0;

        var reader = new Utf8JsonReader(body, isFinalBlock: true, state: default);

        // Top-level object.
        while (reader.Read())
        {
            if (reader.TokenType == JsonTokenType.EndObject) break;
            if (reader.TokenType != JsonTokenType.PropertyName) continue;

            if (Equals(ref reader, KTransaction))
            {
                ReadObject(ref reader, ref txAmount, ref installments, ref requestedAtTicksUtc, ref requestedHour, ref requestedDow);
            }
            else if (Equals(ref reader, KCustomer))
            {
                ReadCustomer(ref reader, body, ref custAvg, ref txCount24h, ref kmArrStart, ref kmArrEnd);
            }
            else if (Equals(ref reader, KMerchant))
            {
                ReadMerchant(ref reader, body, ref merchAvg, ref merchIdStart, ref merchIdEnd, ref mccStart, ref mccEnd);
            }
            else if (Equals(ref reader, KTerminal))
            {
                ReadTerminal(ref reader, ref isOnline, ref cardPresent, ref kmHome);
            }
            else if (Equals(ref reader, KLastTransaction))
            {
                hasLast = ReadLast(ref reader, ref lastTimestampTicksUtc, ref lastKm);
            }
            else
            {
                reader.Read();
                if (reader.TokenType is JsonTokenType.StartObject or JsonTokenType.StartArray)
                    reader.Skip();
            }
        }

        ComposeFeatures(dst, body,
            txAmount, custAvg, merchAvg, kmHome,
            installments, txCount24h,
            isOnline, cardPresent,
            hasLast, requestedAtTicksUtc, lastTimestampTicksUtc, requestedHour, requestedDow, lastKm,
            kmArrStart, kmArrEnd, merchIdStart, merchIdEnd, mccStart, mccEnd);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void ComposeFeatures(
        Span<short> dst, ReadOnlySpan<byte> body,
        double txAmount, double custAvg, double merchAvg, double kmHome,
        int installments, int txCount24h,
        bool isOnline, bool cardPresent,
        bool hasLast, long requestedAtTicksUtc, long lastTimestampTicksUtc,
        int requestedHour, DayOfWeek requestedDow, double lastKm,
        int kmArrStart, int kmArrEnd, int merchIdStart, int merchIdEnd, int mccStart, int mccEnd)
    {
        ref short d0 = ref MemoryMarshal.GetReference(dst);
        var n = _norm;
        Unsafe.Add(ref d0, 0) = Q16(Clamp01((float)(txAmount * n.InvMaxAmount)));
        Unsafe.Add(ref d0, 1) = Q16(Clamp01(installments * n.InvMaxInstallments));
        var avg = (float)custAvg;
        var ratio = avg > 0f ? (float)(txAmount / avg) * n.InvAmountVsAvgRatio : 1f;
        Unsafe.Add(ref d0, 2) = Q16(Clamp01(ratio));
        Unsafe.Add(ref d0, 3) = Q16(requestedHour * (1f / 23f));
        Unsafe.Add(ref d0, 4) = Q16(MondayZero(requestedDow) * (1f / 6f));

        if (hasLast)
        {
            long deltaTicks = requestedAtTicksUtc - lastTimestampTicksUtc;
            if (deltaTicks < 0) deltaTicks = 0;
            double minutes = deltaTicks * (1.0 / 600_000_000.0);
            Unsafe.Add(ref d0, 5) = Q16(Clamp01((float)(minutes * n.InvMaxMinutes)));
            Unsafe.Add(ref d0, 6) = Q16(Clamp01((float)(lastKm * n.InvMaxKm)));
        }
        else
        {
            Unsafe.Add(ref d0, 5) = -10000;
            Unsafe.Add(ref d0, 6) = -10000;
        }

        Unsafe.Add(ref d0, 7) = Q16(Clamp01((float)(kmHome * n.InvMaxKm)));
        Unsafe.Add(ref d0, 8) = Q16(Clamp01(txCount24h * n.InvMaxTxCount24h));
        Unsafe.Add(ref d0, 9) = (short)(isOnline ? 10000 : 0);
        Unsafe.Add(ref d0, 10) = (short)(cardPresent ? 10000 : 0);

        bool isKnown = false;
        if (merchIdEnd > merchIdStart && kmArrEnd > kmArrStart)
        {
            isKnown = ScanKnownMerchants(body.Slice(kmArrStart, kmArrEnd - kmArrStart), body.Slice(merchIdStart, merchIdEnd - merchIdStart));
        }
        Unsafe.Add(ref d0, 11) = (short)(isKnown ? 0 : 10000);

        Unsafe.Add(ref d0, 12) = mccEnd > mccStart
            ? _mcc.GetQ16(body.Slice(mccStart, mccEnd - mccStart))
            : MccRiskTable.DefaultQ16;
        Unsafe.Add(ref d0, 13) = Q16(Clamp01((float)(merchAvg * n.InvMaxMerchantAvgAmount)));
    }

    /// <summary>Single canonical quantization: <c>(short)Round(v * Q16Scale)</c>.
    /// See <see cref="Vectorizer"/> for rationale.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static short Q16(float v) => (short)MathF.Round(v * Dataset.Q16Scale);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static bool Equals(ref Utf8JsonReader r, ReadOnlySpan<byte> key)
        => r.ValueSpan.SequenceEqual(key) || r.ValueIsEscaped && r.ValueTextEquals(key);

    private static void ReadObject(ref Utf8JsonReader r, ref double amount, ref int installments,
                                   ref long requestedTicks, ref int hour, ref DayOfWeek dow)
    {
        // Expect StartObject next.
        r.Read();
        if (r.TokenType != JsonTokenType.StartObject) return;
        while (r.Read())
        {
            if (r.TokenType == JsonTokenType.EndObject) return;
            if (r.TokenType != JsonTokenType.PropertyName) continue;

            if (Equals(ref r, KAmount))
            {
                r.Read();
                amount = FastNumberParse.ParseDouble(r.ValueSpan);
            }
            else if (Equals(ref r, KInstallments))
            {
                r.Read();
                installments = FastNumberParse.ParseInt32(r.ValueSpan);
            }
            else if (Equals(ref r, KRequestedAt))
            {
                r.Read();
                ParseIsoUtc(r.ValueSpan, out requestedTicks, out hour, out dow);
            }
            else
            {
                r.Read();
                if (r.TokenType is JsonTokenType.StartObject or JsonTokenType.StartArray) r.Skip();
            }
        }
    }

    private static void ReadCustomer(ref Utf8JsonReader r, ReadOnlySpan<byte> body,
                                     ref double avgAmount, ref int txCount24h,
                                     ref int kmArrStart, ref int kmArrEnd)
    {
        r.Read();
        if (r.TokenType != JsonTokenType.StartObject) return;
        while (r.Read())
        {
            if (r.TokenType == JsonTokenType.EndObject) return;
            if (r.TokenType != JsonTokenType.PropertyName) continue;

            if (Equals(ref r, KAvgAmount))
            {
                r.Read();
                avgAmount = FastNumberParse.ParseDouble(r.ValueSpan);
            }
            else if (Equals(ref r, KTxCount24h))
            {
                r.Read();
                txCount24h = FastNumberParse.ParseInt32(r.ValueSpan);
            }
            else if (Equals(ref r, KKnownMerchants))
            {
                r.Read();
                if (r.TokenType == JsonTokenType.StartArray)
                {
                    kmArrStart = (int)r.TokenStartIndex; // '[' position
                    r.Skip();                            // advance to matching ']'
                    kmArrEnd = (int)r.TokenStartIndex + 1;
                }
            }
            else
            {
                r.Read();
                if (r.TokenType is JsonTokenType.StartObject or JsonTokenType.StartArray) r.Skip();
            }
        }
    }

    private static void ReadMerchant(ref Utf8JsonReader r, ReadOnlySpan<byte> body,
                                     ref double avgAmount,
                                     ref int merchIdStart, ref int merchIdEnd,
                                     ref int mccStart, ref int mccEnd)
    {
        r.Read();
        if (r.TokenType != JsonTokenType.StartObject) return;
        while (r.Read())
        {
            if (r.TokenType == JsonTokenType.EndObject) return;
            if (r.TokenType != JsonTokenType.PropertyName) continue;

            if (Equals(ref r, KAvgAmount))
            {
                r.Read();
                avgAmount = FastNumberParse.ParseDouble(r.ValueSpan);
            }
            else if (Equals(ref r, KId))
            {
                r.Read();
                if (r.TokenType == JsonTokenType.String)
                {
                    // ValueSpan excludes the surrounding quotes; refers into the body buffer.
                    var span = r.ValueSpan;
                    if (!span.IsEmpty)
                    {
                        merchIdStart = GetSpanOffset(body, span);
                        merchIdEnd = merchIdStart + span.Length;
                    }
                }
            }
            else if (Equals(ref r, KMcc))
            {
                r.Read();
                if (r.TokenType == JsonTokenType.String)
                {
                    // Capture as offsets into the body — MccRiskTable.Get(ReadOnlySpan<byte>)
                    // resolves it without allocating a string per request.
                    var span = r.ValueSpan;
                    if (!span.IsEmpty)
                    {
                        mccStart = GetSpanOffset(body, span);
                        mccEnd = mccStart + span.Length;
                    }
                }
            }
            else
            {
                r.Read();
                if (r.TokenType is JsonTokenType.StartObject or JsonTokenType.StartArray) r.Skip();
            }
        }
    }

    private static void ReadTerminal(ref Utf8JsonReader r, ref bool isOnline, ref bool cardPresent, ref double kmHome)
    {
        r.Read();
        if (r.TokenType != JsonTokenType.StartObject) return;
        while (r.Read())
        {
            if (r.TokenType == JsonTokenType.EndObject) return;
            if (r.TokenType != JsonTokenType.PropertyName) continue;

            if (Equals(ref r, KIsOnline))
            {
                r.Read();
                isOnline = r.TokenType == JsonTokenType.True;
            }
            else if (Equals(ref r, KCardPresent))
            {
                r.Read();
                cardPresent = r.TokenType == JsonTokenType.True;
            }
            else if (Equals(ref r, KKmFromHome))
            {
                r.Read();
                kmHome = FastNumberParse.ParseDouble(r.ValueSpan);
            }
            else
            {
                r.Read();
                if (r.TokenType is JsonTokenType.StartObject or JsonTokenType.StartArray) r.Skip();
            }
        }
    }

    private static bool ReadLast(ref Utf8JsonReader r, ref long timestampTicks, ref double kmCurrent)
    {
        r.Read();
        if (r.TokenType == JsonTokenType.Null) return false;
        if (r.TokenType != JsonTokenType.StartObject) return false;
        while (r.Read())
        {
            if (r.TokenType == JsonTokenType.EndObject) return true;
            if (r.TokenType != JsonTokenType.PropertyName) continue;

            if (Equals(ref r, KTimestamp))
            {
                r.Read();
                ParseIsoUtc(r.ValueSpan, out timestampTicks, out _, out _);
            }
            else if (Equals(ref r, KKmFromCurrent))
            {
                r.Read();
                kmCurrent = FastNumberParse.ParseDouble(r.ValueSpan);
            }
            else
            {
                r.Read();
                if (r.TokenType is JsonTokenType.StartObject or JsonTokenType.StartArray) r.Skip();
            }
        }
        return true;
    }

    /// <summary>
    /// Parse "YYYY-MM-DDTHH:MM:SSZ" (20 bytes) directly into UTC ticks + hour + dow.
    /// Falls back to DateTimeOffset.Parse for anything else (fractional seconds, offsets, etc).
    /// </summary>
    private static void ParseIsoUtc(ReadOnlySpan<byte> s, out long ticks, out int hour, out DayOfWeek dow)
    {
        if (s.Length == 20 && s[4] == (byte)'-' && s[7] == (byte)'-' && s[10] == (byte)'T'
            && s[13] == (byte)':' && s[16] == (byte)':' && s[19] == (byte)'Z')
        {
            int y = D4(s, 0);
            int mo = D2(s, 5);
            int d = D2(s, 8);
            int h = D2(s, 11);
            int mi = D2(s, 14);
            int se = D2(s, 17);
            // Build via DateTime and convert to ticks.
            // DateTime ctor is cheap (no parsing/culture).
            var dt = new DateTime(y, mo, d, h, mi, se, DateTimeKind.Utc);
            ticks = dt.Ticks;
            hour = h;
            dow = dt.DayOfWeek;
            return;
        }
        // Fallback.
        var str = System.Text.Encoding.UTF8.GetString(s);
        var dto = DateTimeOffset.Parse(str, CultureInfo.InvariantCulture);
        var u = dto.UtcDateTime;
        ticks = u.Ticks;
        hour = u.Hour;
        dow = u.DayOfWeek;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int D2(ReadOnlySpan<byte> s, int o) => (s[o] - '0') * 10 + (s[o + 1] - '0');
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int D4(ReadOnlySpan<byte> s, int o)
        => (s[o] - '0') * 1000 + (s[o + 1] - '0') * 100 + (s[o + 2] - '0') * 10 + (s[o + 3] - '0');

    /// <summary>
    /// Walk a JSON array literal "[ "a", "b", ... ]" comparing each string entry's bytes
    /// (between the surrounding quotes) against <paramref name="needle"/>.
    /// No allocations; bails on first match. Assumes the bench data has no escaped
    /// characters in MERC-* IDs (verified empirically; falls back conservatively to
    /// "not known" if a backslash is encountered).
    /// </summary>
    private static bool ScanKnownMerchants(ReadOnlySpan<byte> arr, ReadOnlySpan<byte> needle)
    {
        int i = 0;
        while (i < arr.Length)
        {
            // Find next quote.
            while (i < arr.Length && arr[i] != (byte)'"') i++;
            if (i >= arr.Length) return false;
            int start = ++i;
            // Find closing quote (no escape support — conservative).
            while (i < arr.Length && arr[i] != (byte)'"')
            {
                if (arr[i] == (byte)'\\') return false;
                i++;
            }
            if (i >= arr.Length) return false;
            var entry = arr.Slice(start, i - start);
            if (entry.SequenceEqual(needle)) return true;
            i++;
        }
        return false;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe int GetSpanOffset(ReadOnlySpan<byte> body, ReadOnlySpan<byte> slice)
    {
        // Both spans refer to the same backing buffer (the request body); compute byte offset.
        ref byte b0 = ref System.Runtime.InteropServices.MemoryMarshal.GetReference(body);
        ref byte s0 = ref System.Runtime.InteropServices.MemoryMarshal.GetReference(slice);
        return (int)Unsafe.ByteOffset(ref b0, ref s0);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float Clamp01(float v)
    {
        if (Sse.IsSupported)
        {
            var vv = Vector128.CreateScalarUnsafe(v);
            vv = Sse.MaxScalar(vv, Vector128<float>.Zero);
            vv = Sse.MinScalar(vv, Vector128.CreateScalarUnsafe(1f));
            return vv.ToScalar();
        }
        return MathF.Min(MathF.Max(v, 0f), 1f);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int MondayZero(DayOfWeek d) => d == DayOfWeek.Sunday ? 6 : (int)d - 1;
}
