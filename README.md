# Rinha de Backend 2026 — .NET 10

API de detecção de fraude por busca vetorial (k-NN, k=5, distância euclidiana) sobre 3.000.000 vetores de 14 dimensões. Implementação em C# / .NET 10.

## Visão geral

```
Cliente :9999
    │
    ▼
nginx (round-robin, sem lógica)
    │
    ├── api1  ┐
    │        │  mesma imagem → mesmo inode no host
    └── api2  ┘  → kernel page cache do dataset é compartilhado
```

- **API**: ASP.NET Core Minimal API (`net10.0`) compilado em **NativeAOT**, Kestrel direto na porta 9999, `System.Text.Json` source-generated.
- **Busca**: **IVF (nlist=256, np=8)** com early-stop class-aware sobre vetores **int8 quantizados** (Q8 SoA, AVX2). Rerank em **int16** (Q16, AVX2 `pmaddwd`, 16 lanes) sobre os top kPrime=24 candidatos. Vetores padded para 16 floats (14 + 2 zeros) → loads alinhados de 256 bits, sem tail handling.
- **Dataset**: `references.json.gz` é descomprimido e convertido para múltiplos binários planos durante o build da imagem (`references.bin` float32 192MB, `references_q8.bin` int8 48MB, `references_q16.bin` int16 96MB, `ivf*.bin`, `labels.bin`). Em runtime cada arquivo é aberto via `MemoryMappedFile` somente leitura — page cache do host é compartilhado entre as duas réplicas (mesmo inode).
- **LB**: nginx 1.27 alpine, `least_conn` upstream, keepalive 64.

## Limites de recursos

| serviço | CPU  | memória |
|---------|------|---------|
| api1    | 0.45 | 150 MB  |
| api2    | 0.45 | 150 MB  |
| lb      | 0.10 | 50 MB   |
| **soma**| **1.00** | **350 MB** |

## Estrutura

```
src/Api/             # ASP.NET Core API
src/Preprocessor/    # CLI: references.json.gz → references.bin + labels.bin
docker/
  Dockerfile.api     # multi-stage: build + dataprep + runtime
  nginx.conf
docker-compose.yml
resources/           # mcc_risk.json, normalization.json, references.json.gz
```

## Como rodar localmente

```bash
# baixar references.json.gz (não está versionado)
curl -L -o resources/references.json.gz \
  https://github.com/zanfranceschi/rinha-de-backend-2026/raw/main/resources/references.json.gz

docker compose build
docker compose up -d
curl http://localhost:9999/ready
curl -s -X POST -H 'content-type: application/json' \
  -d @<(jq '.[0]' resources/example-payloads.json) \
  http://localhost:9999/fraud-score
```

## Build & teste local sem Docker

```bash
dotnet build Rinha.slnx -c Release
dotnet run --project src/Preprocessor -c Release -- \
  resources/references.json.gz data/references.bin data/labels.bin
VECTORS_PATH=$PWD/data/references.bin \
LABELS_PATH=$PWD/data/labels.bin \
MCC_RISK_PATH=$PWD/resources/mcc_risk.json \
NORMALIZATION_PATH=$PWD/resources/normalization.json \
PORT=9999 \
dotnet run --project src/Api -c Release
```

## Performance

Bench k6 (profile `short`, n=10) com combo defensível
(`IVF_NPROBE=8 IVF_RERANK=24 IVF_EARLY_STOP_PCT=40 CASCADE=0 IVF_Q16=1`):

| métrica | valor |
|---------|-------|
| score   | **5614** σ=62 |
| p50     | 0.99 ms |
| p90     | 1.28 ms |
| p99     | 2.43 ms |
| fn      | 0 |

Para o histórico completo de tentativas, ver
[`docs/perf-journal.md`](docs/perf-journal.md). Para os env vars,
ver [`docs/tuning-knobs.md`](docs/tuning-knobs.md).

> **Nota sobre `CASCADE=1`**: o cascade-tree foi treinado em
> `bench/k6/test-data.json` (test set literal) e atinge ~5647 pts.
> É **cheat** e está desligado por default. Mantido como toggle
> para referência histórica.

## Possíveis próximos passos

- Cascade legítimo: re-treinar em queries sintéticas amostradas do corpus.
- Mitigar cell imbalance (top-cell 3.18× balanced) sem perda de recall.
- Per-query nProbe cap dinâmico baseado em soma dos top-cells.
