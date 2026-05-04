using System.Buffers;
using System.Buffers.Binary;
using System.IO.Compression;
using System.Text.Json;

if (args.Length < 3 || args.Length > 4)
{
    Console.Error.WriteLine("Usage: Rinha.Preprocessor <references.json.gz> <out-vectors.bin> <out-labels.bin> [<out-vectors-q8.bin>]");
    return 1;
}

var inputPath = args[0];
var vectorsPath = args[1];
var labelsPath = args[2];
var vectorsQ8Path = args.Length == 4 ? args[3] : null;

const int Dimensions = 14;
const int PaddedDimensions = 16; // align to Vector256<float> (8 lanes)
const int RowBytes = PaddedDimensions * sizeof(float);
const int Q8RowBytes = PaddedDimensions; // 1 byte per dim, padded to 16

if (!File.Exists(inputPath))
{
    Console.Error.WriteLine($"Input not found: {inputPath}");
    return 2;
}

Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(vectorsPath))!);
Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(labelsPath))!);
if (vectorsQ8Path is not null)
    Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(vectorsQ8Path))!);

var sw = System.Diagnostics.Stopwatch.StartNew();

await using var fs = File.OpenRead(inputPath);
await using var gz = new GZipStream(fs, CompressionMode.Decompress);
await using var bufferedGz = new BufferedStream(gz, 1 << 20);

await using var vectorsOut = new FileStream(vectorsPath, FileMode.Create, FileAccess.Write, FileShare.None, 1 << 20, FileOptions.SequentialScan);
await using var labelsOut = new FileStream(labelsPath, FileMode.Create, FileAccess.Write, FileShare.None, 1 << 20, FileOptions.SequentialScan);
FileStream? q8Out = vectorsQ8Path is null
    ? null
    : new FileStream(vectorsQ8Path, FileMode.Create, FileAccess.Write, FileShare.None, 1 << 20, FileOptions.SequentialScan);

long count;
try
{
    count = StreamConvert(bufferedGz, vectorsOut, labelsOut, q8Out);
}
finally
{
    if (q8Out is not null) await q8Out.DisposeAsync();
}

await vectorsOut.FlushAsync();
await labelsOut.FlushAsync();

sw.Stop();
Console.Error.WriteLine($"Done: {count:N0} vectors in {sw.Elapsed}.");
Console.Error.WriteLine($"Vectors:    {vectorsPath} ({new FileInfo(vectorsPath).Length:N0} bytes)");
Console.Error.WriteLine($"Labels:     {labelsPath} ({new FileInfo(labelsPath).Length:N0} bytes)");
if (vectorsQ8Path is not null)
    Console.Error.WriteLine($"Vectors q8: {vectorsQ8Path} ({new FileInfo(vectorsQ8Path).Length:N0} bytes)");
return 0;

static long StreamConvert(Stream input, Stream vectorsOut, Stream labelsOut, Stream? q8Out)
{
    const int InitialBufferSize = 1 << 20;
    var buffer = ArrayPool<byte>.Shared.Rent(InitialBufferSize);
    long count = 0;

    try
    {
        int bytesInBuffer = 0;
        bool isFinalBlock = false;
        var state = new JsonReaderState(new JsonReaderOptions
        {
            CommentHandling = JsonCommentHandling.Skip,
            AllowTrailingCommas = true,
        });

        // 0=expect '[', 1=expect '{' or ']', 2=inside obj expecting prop or '}',
        // 3=after "vector" prop, expect '[', 4=inside vec, expect number or ']',
        // 5=after "label" prop, expect string,
        // 6=after unknown prop, expect any value (and skip),
        // -1=done
        int parserState = 0;
        Span<float> vec = stackalloc float[Dimensions];
        int vecPos = 0;
        byte? currentLabel = null;
        bool haveVector = false;
        var rowBuf = new byte[RowBytes];
        var q8Buf  = new byte[Q8RowBytes]; // sbyte semantically; written as bytes

        while (parserState != -1)
        {
            int read = input.Read(buffer, bytesInBuffer, buffer.Length - bytesInBuffer);
            if (read == 0) isFinalBlock = true;
            bytesInBuffer += read;

            var span = buffer.AsSpan(0, bytesInBuffer);
            var reader = new Utf8JsonReader(span, isFinalBlock, state);

            bool needMoreData = false;
            while (!needMoreData && parserState != -1)
            {
                if (!reader.Read())
                {
                    needMoreData = true;
                    break;
                }

                switch (parserState)
                {
                    case 0:
                        Expect(reader.TokenType, JsonTokenType.StartArray, "root");
                        parserState = 1;
                        break;

                    case 1:
                        if (reader.TokenType == JsonTokenType.EndArray) { parserState = -1; }
                        else if (reader.TokenType == JsonTokenType.StartObject)
                        {
                            vecPos = 0;
                            currentLabel = null;
                            haveVector = false;
                            parserState = 2;
                        }
                        else throw new InvalidDataException($"Unexpected token at root array: {reader.TokenType}");
                        break;

                    case 2:
                        if (reader.TokenType == JsonTokenType.EndObject)
                        {
                            if (!haveVector || currentLabel is null)
                                throw new InvalidDataException($"Record {count}: missing vector or label");
                            Array.Clear(rowBuf, Dimensions * sizeof(float), (PaddedDimensions - Dimensions) * sizeof(float));
                            for (int i = 0; i < Dimensions; i++)
                                BinaryPrimitives.WriteSingleLittleEndian(rowBuf.AsSpan(i * sizeof(float), sizeof(float)), vec[i]);
                            vectorsOut.Write(rowBuf, 0, RowBytes);
                            if (q8Out is not null)
                            {
                                // Quantize to int8: round(f * 127), clamp to [-128, 127].
                                // f ∈ [0,1] → [0,127]; sentinel f = -1 → -127 (natural mapping).
                                Array.Clear(q8Buf, 0, Q8RowBytes);
                                for (int i = 0; i < Dimensions; i++)
                                {
                                    int q = (int)MathF.Round(vec[i] * 127f);
                                    if (q > 127) q = 127;
                                    else if (q < -128) q = -128;
                                    q8Buf[i] = (byte)(sbyte)q;
                                }
                                q8Out.Write(q8Buf, 0, Q8RowBytes);
                            }
                            labelsOut.WriteByte(currentLabel.Value);
                            count++;
                            if (count % 500_000 == 0) Console.Error.WriteLine($"...processed {count:N0}");
                            parserState = 1;
                        }
                        else if (reader.TokenType == JsonTokenType.PropertyName)
                        {
                            if (reader.ValueTextEquals("vector")) parserState = 3;
                            else if (reader.ValueTextEquals("label")) parserState = 5;
                            else parserState = 6;
                        }
                        else throw new InvalidDataException($"Record {count}: expected property, got {reader.TokenType}");
                        break;

                    case 3:
                        Expect(reader.TokenType, JsonTokenType.StartArray, $"record {count} vector");
                        vecPos = 0;
                        parserState = 4;
                        break;

                    case 4:
                        if (reader.TokenType == JsonTokenType.EndArray)
                        {
                            if (vecPos != Dimensions)
                                throw new InvalidDataException($"Record {count}: expected {Dimensions} dims, got {vecPos}");
                            haveVector = true;
                            parserState = 2;
                        }
                        else if (reader.TokenType == JsonTokenType.Number)
                        {
                            if (vecPos >= Dimensions)
                                throw new InvalidDataException($"Record {count}: vector too long");
                            vec[vecPos++] = reader.GetSingle();
                        }
                        else throw new InvalidDataException($"Record {count}: bad token in vector: {reader.TokenType}");
                        break;

                    case 5:
                        Expect(reader.TokenType, JsonTokenType.String, $"record {count} label");
                        if (reader.ValueTextEquals("fraud")) currentLabel = 1;
                        else if (reader.ValueTextEquals("legit")) currentLabel = 0;
                        else throw new InvalidDataException($"Record {count}: unknown label");
                        parserState = 2;
                        break;

                    case 6:
                        // Any value; if it's a start, skip subtree.
                        if (reader.TokenType == JsonTokenType.StartObject || reader.TokenType == JsonTokenType.StartArray)
                        {
                            if (!reader.TrySkip())
                            {
                                needMoreData = true;
                                break;
                            }
                        }
                        // else value is single token already consumed.
                        parserState = 2;
                        break;
                }
            }

            state = reader.CurrentState;
            int consumed = (int)reader.BytesConsumed;

            if (consumed > 0)
            {
                if (consumed < bytesInBuffer)
                    Buffer.BlockCopy(buffer, consumed, buffer, 0, bytesInBuffer - consumed);
                bytesInBuffer -= consumed;
            }

            if (parserState == -1) break;

            if (isFinalBlock && consumed == 0)
                throw new InvalidDataException($"Unexpected end of input at record {count}");

            if (bytesInBuffer == buffer.Length)
            {
                var bigger = ArrayPool<byte>.Shared.Rent(buffer.Length * 2);
                Buffer.BlockCopy(buffer, 0, bigger, 0, bytesInBuffer);
                ArrayPool<byte>.Shared.Return(buffer);
                buffer = bigger;
            }
        }

        return count;
    }
    finally
    {
        ArrayPool<byte>.Shared.Return(buffer);
    }

    static void Expect(JsonTokenType actual, JsonTokenType expected, string ctx)
    {
        if (actual != expected) throw new InvalidDataException($"{ctx}: expected {expected}, got {actual}");
    }
}
