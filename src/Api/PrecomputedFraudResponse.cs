using System.Text.Json;
using Microsoft.AspNetCore.Http;

namespace Rinha.Api;

// J11d: pré-serializa as 6 respostas possíveis (fraud_score = N/5 para N ∈ 0..5).
// Substitui Results.Json(new FraudResponse(...)) — evita alocação de FraudResponse,
// passagem por JsonSerializer e formatação de float a cada request.
//
// Implementado como sealed class (não struct) porque o source generator de
// minimal API gera comparações `result == null` que não compilam contra struct.
// Cacheamos as 6 instâncias estáticas — uma por valor possível — então não há
// alocação por request mesmo sendo classe.
public sealed class PrecomputedFraudResponse : IResult
{
    /// <summary>
    /// Approval threshold: <c>approved = score &lt; ApprovalThreshold</c>. With k=5 the
    /// six discrete scores are {0, .2, .4, .6, .8, 1}, and 0.6 is the smallest value
    /// that means &quot;majority of nearest neighbours are fraud&quot;. Shared with
    /// <see cref="Rinha.Api.Scorers.HybridIvfQ16Scorer"/> so any change here
    /// automatically updates the borderline-trigger logic.
    /// </summary>
    public const float ApprovalThreshold = 0.6f;

    /// <summary>Granularity of the score (1 / K, where K is the kNN k = 5).</summary>
    public const float ScoreStep = 0.2f;

    private static readonly byte[][] Bodies = BuildBodies();
    private static readonly PrecomputedFraudResponse[] Instances = BuildInstances();

    private readonly int _index;

    private PrecomputedFraudResponse(int index) => _index = index;

    public static PrecomputedFraudResponse FromScore(float score)
    {
        // score deve ser N/5 com N ∈ 0..5; clamp + round defensivos.
        int n = (int)MathF.Round(score * 5f);
        if (n < 0) n = 0;
        else if (n > 5) n = 5;
        return Instances[n];
    }

    public Task ExecuteAsync(HttpContext httpContext)
    {
        var body = Bodies[_index];
        var resp = httpContext.Response;
        resp.StatusCode = 200;
        resp.ContentType = "application/json";
        resp.ContentLength = body.Length;
        return resp.Body.WriteAsync(body, 0, body.Length);
    }

    private static byte[][] BuildBodies()
    {
        var arr = new byte[6][];
        for (int n = 0; n <= 5; n++)
        {
            float score = n / 5f;
            var resp = new FraudResponse(score < ApprovalThreshold, score);
            arr[n] = JsonSerializer.SerializeToUtf8Bytes(resp, AppJsonContext.Default.FraudResponse);
        }
        return arr;
    }

    private static PrecomputedFraudResponse[] BuildInstances()
    {
        var arr = new PrecomputedFraudResponse[6];
        for (int n = 0; n <= 5; n++) arr[n] = new PrecomputedFraudResponse(n);
        return arr;
    }
}
