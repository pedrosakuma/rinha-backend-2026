# DateParseBench

Microbenchmark comparing three implementations of parsing the canonical ISO 8601 UTC
timestamp `"YYYY-MM-DDTHH:MM:SSZ"` (exactly 20 ASCII bytes) into a .NET `long` (UTC ticks).

Context: `src/Api/JsonVectorizer.cs::ParseIsoUtc` is called for every date field in each
incoming API request (up to 2 calls per request). Even single-digit nanosecond savings
compound at high RPS.

---

## How to run

```bash
cd tools/DateParseBench
dotnet run -c Release
```

> **Note:** This project targets `net10.0` and runs on the **JIT**, _not_ NativeAOT.
> The production API (`src/Api/`) is NativeAOT — absolute nanosecond numbers will differ.
> Use the *Ratio* column for relative comparisons between variants.

To see the generated assembly for a method, add `[DisassemblyDiagnoser]` to the
benchmark class and re-run.

---

## Variants

### `Scalar` *(baseline)*

Mirrors the current `ParseIsoUtc` exactly:
- `D4(s, 0)` / `D2(s, off)` — reads bytes one at a time, multiplies by positional
  weight (1000, 100, 10, 1).
- Passes the six fields to `new DateTime(y, mo, d, h, mi, se, DateTimeKind.Utc)` and
  returns `.Ticks`.

### `UlongSwar` — SWAR with two `ulong` reads

**SWAR** = *SIMD Within A Register*: exploit the fact that a 64-bit register can hold
8 bytes and process them in parallel with integer instructions.

1. **Two loads** via `Unsafe.ReadUnaligned<ulong>`:
   - `u0` ← bytes 0–7  (`"YYYY-MM-"`)
   - `u1` ← bytes 8–15 (`"DDTHH:MM"` with 'T' and ':' as separators)
2. Subtract `0x3030…30` from both registers simultaneously — all 8 bytes per register
   get `- '0'` in one instruction.
3. Shift + mask to pick out each digit field; combine with multiplications.
4. Bytes 17–18 (`SS`) are read scalar (the 16-byte load doesn't reach them).
5. Ticks are computed without `new DateTime()` using **Howard Hinnant's
   `days_from_civil` algorithm** (pure integer arithmetic, no heap allocation):
   ```
   unixDays = era*146097 + doe - 719468
   ticks    = (unixDays*86400 + h*3600 + mi*60 + s) * 10_000_000
            + DateTime.UnixEpoch.Ticks
   ```

**Key win:** 2 memory reads instead of 10+ independent byte loads; separator bytes are
discarded via masks rather than conditional checks.

### `Vector128Swar` — SWAR with a 128-bit SIMD load (SSSE3)

1. **One `Vector128<byte>` load** (`Vector128.LoadUnsafe`) covers bytes 0–15 in a
   single `movdqu` instruction.
2. **`pshufb`** (`Ssse3.Shuffle`) gathers the 12 digit bytes from their scattered
   positions into a contiguous arrangement:
   ```
   [y3,y2,y1,y0, m1,m0, d1,d0, h1,h0, mi1,mi0, ...]
   ```
3. **`pmaddubsw`** (`Ssse3.MultiplyAddAdjacent`) multiplies each byte by a coefficient
   vector `[10,1, 10,1, …]` and adds adjacent products in one instruction:
   ```
   pairs[0] = y_thousands*10 + y_hundreds   → 20  (for "2026")
   pairs[1] = y_tens*10      + y_units       → 26
   pairs[2] = month,  pairs[3] = day,  pairs[4] = hour,  pairs[5] = minute
   ```
4. Year = `pairs[0]*100 + pairs[1]`.
5. Seconds tail (bytes 17–18) read scalar; same Hinnant ticks formula.

Falls back to element-wise extraction if SSSE3 is not available at runtime.

---

## Correctness guarantee

`GlobalSetup` validates that all three implementations produce identical ticks for the
test timestamp (`"2026-05-07T14:23:45Z"`) against `new DateTime(...).Ticks`. If any
variant disagrees, it throws with a diff before the benchmark runs.

---

## Expected behaviour

All three allocate **0 bytes** (shown by `MemoryDiagnoser`). The Ratio column is the
interesting metric — see the benchmark output for your machine.
