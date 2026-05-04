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

- **API**: ASP.NET Core Minimal API (`net10.0`), Kestrel direto na porta 9999, `System.Text.Json` source-generated.
- **Busca**: brute force k-NN com SIMD (`Vector256<float>`). Cada vetor de referência é gravado em 16 floats (14 + 2 zeros de padding) → exatamente 2 loads de 256 bits, sem tail handling.
- **Dataset**: `references.json.gz` é descomprimido e convertido para um arquivo binário plano (`references.bin`, 192 MB row-major float32 + `labels.bin`, 1 byte/registro) durante o build da imagem. Em runtime o arquivo é aberto via `MemoryMappedFile` somente leitura.
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

## Performance (smoke test local, cliente sequencial)

| métrica | valor |
|---------|-------|
| p50     | ~20 ms |
| p99     | ~85 ms |
| memória total | ~60 MB |

## Possíveis próximos passos

Ideias para iterar sobre essa baseline (não há compromisso de implementar):

- Particionar a busca em múltiplas threads por requisição.
- Avaliar IVF/clustering para reduzir o número de candidatos.
- Migrar para NativeAOT.
- Tuning fino de Kestrel/`HttpJson`.
