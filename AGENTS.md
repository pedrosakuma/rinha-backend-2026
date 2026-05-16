# AGENTS.md

Instruções para agentes/automação que contribuírem para este repositório (submissão à Rinha de Backend 2026).

## Regras de competição (CRÍTICAS — auditáveis)

Fonte oficial: https://github.com/zanfranceschi/rinha-de-backend-2026/blob/main/docs/br/FAQ.md

**O que NÃO é permitido:**

1. **Usar os payloads do teste como referência ou para fazer lookup de fraudes.**
   - ❌ Ler `test-data.json` (ou qualquer subset dele) **offline** para construir tabelas/caches/índices que são embutidos na imagem ou no código.
   - ❌ Pré-computar respostas a partir do `expected_fraud_score` do test-data.
   - ❌ Treinar classificadores/buckets cuja origem das amostras é o test-data.
   - ✅ É permitido usar `references.bin` (dataset oficial de treinamento, 3M vetores rotulados) como fonte de qualquer pré-computação, índice ou tabela.
   - ✅ Em runtime, é permitido cachear/aprender com **as próprias respostas computadas legitimamente** via KNN/ANN sobre as queries que chegam ao servidor (online learning baseado nas respostas que nós mesmos emitimos).

2. Load balancer só distribui requisições — não pode aplicar lógica de detecção de fraude.

3. Repositório e imagens Docker devem ser públicos.

**Antes de adicionar qualquer nova stage/cache/lookup table, pergunte: "essa estrutura foi construída lendo `test-data.json`?" Se sim, é violação.**

Histórico relevante: as stages `borderline-residual-purity`, `tail-residual-purity`, `final-residual-purity` foram desabilitadas em wave37 por violarem essa regra (eram treinadas via `BuildResidualPurity.cs` que lê `test-data.json`). Veja eval #4543 (5975 não-compliant) → #4550 (5921 compliant).

## Pipeline de submissão

1. Branch: `submission` (origin). Trabalhamos local em `submission-local`.
2. Imagens: `ghcr.io/pedrosakuma/rinha-backend-2026-api:<wave>-<tag>`.
3. `docker-compose.yml` referencia a imagem via tag e usa `pull_policy: always` em prod (`never` localmente para teste rápido).
4. Após push, abrir issue no repo do organizador para disparar eval:
   ```bash
   gh issue create --repo zanfranceschi/rinha-de-backend-2026 \
     --title "rinha/test pedrosakuma-dotnet" \
     --body  "rinha/test pedrosakuma-dotnet"
   ```
5. Bot `arinhadebackend` comenta com resultado em ~5-30 min (pode estar congestionado: até 4h).

## Comandos

`dotnet` está em `/home/azureuser/.dotnet` — adicione ao PATH.

```bash
# Build
dotnet build src/Api/Api.csproj   -c Release -v q   # ~5s
dotnet build src/Bench/Bench.csproj -c Release -v q # ~7s

# Replay (validação FN/FP)
dotnet run --project src/Bench/Bench.csproj -c Release --no-build -- \
  --replay --test-data=bench/k6/test-data.json --scorer=ivf-blocked --config="NP=1"

# Profile families (per-family p50/p99)
dotnet run --project src/Bench/Bench.csproj -c Release --no-build -- \
  --perf-families --test-data=bench/k6/test-data.json

# Stage histogram (per-stage isolated p99)
dotnet run --project src/Bench/Bench.csproj -c Release --no-build -- \
  --stage-histogram --test-data=bench/k6/test-data.json --repeats=5

# A/B local 3x
bench/run-x3.sh --profile=full --label=<tag> --n=3   # ~7min

# Imagem
docker build -f docker/Dockerfile.api -t ghcr.io/pedrosakuma/rinha-backend-2026-api:<wave> .
docker push ghcr.io/pedrosakuma/rinha-backend-2026-api:<wave>
```

## Convenções de código

- **Style**: comente apenas o que precisa de explicação. Sem comentários ruidosos.
- **Performance**: `[MethodImpl(MethodImplOptions.AggressiveInlining)]` em hot path. Evitar alocação em handler de request.
- **NativeAOT-friendly**: sem reflexão, JsonSerializer source-generated.
- **Compose**: cpuset isolado por container (api1=0,1; api2=2,3; lb=0,2 com overlap intencional).

## Não retomar (waves anteriores derrotadas)

- ❌ Hand-rolled JSON parser (waves 26/27/28a/28b: -3% a -7%).
- ❌ Forward-scan IndexOf parser (-5%).
- ❌ Positional Utf8JsonReader (-3.7%).
- ❌ `DOTNET_GCRetainVM=1` + `GCHeapHardLimit=128MB` com workstation GC + cpuset pequeno (-95 pts em eval).
- ❌ `nlist=1536` (regrediu).
- ❌ `TP=12/14`, `RAW_HTTP_WORKERS>1`, `WORKERS=1` no LB (regrediram).
- ❌ Stages residual_modal_sparse com origem em test-data.json (compliance).
- ❌ FP16 storage para refs (wave38 PoC: 16 fraud-count divergences em 54100, precisão ~10⁻³ insuficiente vs Q16 5×10⁻⁵).
- ❌ VPMADDWD path no IVF block scan: widen per-par (Vector256&lt;long&gt; via vpmovsxdq×2) regrediu -8% p50/p99; sem widen (Vector256&lt;int&gt; acc + widen-at-end) é perf-neutro mas tem 1 FP/54100 por wrap int32 (true sum &gt; 4.3e9 em queries patológicas). Float fmadd é o ótimo local pra esse hot path no Zen3.

## Estado atual (referência)

- Submissão em `submission@HEAD`.
- Cascade: 2 stages (RP1 92.67%, RP2 1.86%) + IVF (5.47%).
- Eval baseline compliant: **5921.11 / p99 1.20ms** (#4550).
