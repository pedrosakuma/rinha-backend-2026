using System.Buffers;
using System.Buffers.Text;
using System.Globalization;
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
        string mccStr = string.Empty;

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
                ReadMerchant(ref reader, body, ref merchAvg, ref merchIdStart, ref merchIdEnd, ref mccStr);
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

        // ---- Compose the 14 features (mirrors Vectorizer.Vectorize) ----
        var n = _norm;
        dst[0] = Clamp01((float)(txAmount / n.MaxAmount));
        dst[1] = Clamp01(installments / n.MaxInstallments);
        var avg = (float)custAvg;
        var ratio = avg > 0f ? (float)(txAmount / avg) / n.AmountVsAvgRatio : 1f;
        dst[2] = Clamp01(ratio);
        dst[3] = requestedHour / 23f;
        dst[4] = MondayZero(requestedDow) / 6f;

        if (hasLast)
        {
            // Both ticks are in 100ns units (DateTime.Ticks). 1 minute = 600,000,000 ticks.
            long deltaTicks = requestedAtTicksUtc - lastTimestampTicksUtc;
            if (deltaTicks < 0) deltaTicks = 0;
            double minutes = deltaTicks / 600_000_000.0;
            dst[5] = Clamp01((float)(minutes / n.MaxMinutes));
            dst[6] = Clamp01((float)(lastKm / n.MaxKm));
        }
        else
        {
            dst[5] = -1f;
            dst[6] = -1f;
        }

        dst[7] = Clamp01((float)(kmHome / n.MaxKm));
        dst[8] = Clamp01(txCount24h / n.MaxTxCount24h);
        dst[9] = isOnline ? 1f : 0f;
        dst[10] = cardPresent ? 1f : 0f;

        // unknown_merchant: scan known_merchants array, byte-compare each entry vs merchant.id slice.
        bool isKnown = false;
        if (merchIdEnd > merchIdStart && kmArrEnd > kmArrStart)
        {
            isKnown = ScanKnownMerchants(body.Slice(kmArrStart, kmArrEnd - kmArrStart), body.Slice(merchIdStart, merchIdEnd - merchIdStart));
        }
        dst[11] = isKnown ? 0f : 1f;

        dst[12] = _mcc.Get(mccStr);
        dst[13] = Clamp01((float)(merchAvg / n.MaxMerchantAvgAmount));

        // Match oracle quantization: references in resources/references.json.gz are stored
        // pre-rounded to 4 decimal places. Round queries the same way so distance
        // comparisons (and tie-breaks) align with the oracle's k-NN ground truth.
        for (int i = 0; i < Dataset.Dimensions; i++) dst[i] = Round4dp(dst[i]);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float Round4dp(float v)
    {
        if (v < 0f) return v; // preserve -1 sentinel
        return MathF.Round(v * 10000f) / 10000f;
    }

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
                amount = r.GetDouble();
            }
            else if (Equals(ref r, KInstallments))
            {
                r.Read();
                installments = r.GetInt32();
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
                avgAmount = r.GetDouble();
            }
            else if (Equals(ref r, KTxCount24h))
            {
                r.Read();
                txCount24h = r.GetInt32();
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
                                     ref string mccStr)
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
                avgAmount = r.GetDouble();
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
                    mccStr = r.GetString() ?? string.Empty;
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
                kmHome = r.GetDouble();
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
                kmCurrent = r.GetDouble();
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
    private static float Clamp01(float v) => v < 0f ? 0f : (v > 1f ? 1f : v);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int MondayZero(DayOfWeek d) => d == DayOfWeek.Sunday ? 6 : (int)d - 1;
}
