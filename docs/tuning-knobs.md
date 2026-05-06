# Tuning Knobs — env vars do scorer

> Referência completa dos toggles runtime. Para o histórico de decisões,
> ver [`perf-journal.md`](perf-journal.md).

## Combo defensível recomendado

```
IVF_NPROBE=8 IVF_RERANK=24 IVF_EARLY_STOP_PCT=40 CASCADE=0 IVF_Q16=1
```

Score n=10: 5614 σ=62 p99=2.43ms fn=0.

## Knobs principais

| Env | Default | Range testado | Efeito |
|-----|--------:|---|---|
| `IVF_NPROBE` | `8` | 4-96 | Cells visitadas por query. Reduzido de 96 quando a hot path passou para `cascade-on`/Q16. |
| `IVF_RERANK` | `64` (compose) / **24 recomendado** | 16-128 | kPrime: candidatos enviados ao rerank float/Q16. Sweet spot 24 cascade-off. |
| `IVF_EARLY_STOP` | `1` | 0/1 | Liga early-stop class-aware (J5). Sempre on em prod. |
| `IVF_EARLY_STOP_PCT` | `72` (compose) / **40 recomendado** | 30-80 | % de cells visitadas antes do checkpoint de early-stop. Cascade-off prefere 40, cheat-mode 72. |
| `CASCADE` | `1` (compose) / **0 recomendado** | 0/1 | Liga cascade decision-tree. **Default atual = cheat** (treinado em test-data.json). Use 0 para combo defensível. |
| `IVF_Q16` | `1` | 0/1 | Rerank em int16 (AVX2 pmaddwd, 16 lanes). Default-on quando `references_q16.bin` existe. |
| `SCORER` | `ivf` | `ivf`, `q8recheck`, `bruteforce` | Algoritmo. Histórico — `ivf` é o único usado em prod. |

## Knobs experimentais (toggles default OFF)

Mantidos para reproduzir investigações; não habilitar em prod sem bench:

| Env | Default | O que faz | Status |
|-----|--------:|---|---|
| `IVF_BBOX_REPAIR` | `0` | J6 — bbox LB exact repair | Rejeitado, em árvore |
| `IVF_SCALAR_ABORT` | `0` | 1=AoS dim-unroll (J10, -489), 2=SoA (J11, -731) | Rejeitado |
| `IVF_DENSITY_ORDER` | `0` | J12a — reorder visited cells por size desc | Rejeitado (-52 noise) |
| `IVF_EARLY_MAJORITY` | `0` | J21 — majority 4/5 early-stop | Opt-in only |
| `IVF_DEADLINE_US` | `0` | J22 — anytime/deadline scan | Opt-in only |
| `IVF_BATCH_WAIT_US` | `0` | L1 — query batching (microbatch wait) | Opt-in only |
| `IVF_PREFETCH` | `0` | L2 — DRAM prefetch | Sem ganho consistente |
| `IVF_BBOX_REPAIR` | `0` | bbox LB exact repair | Rejeitado |
| `IVF_EARLY_STOP_PCT_EARLY` | `0` | early-stop checkpoint mais cedo (separado) | Não usado |
| `LIFO_LIMIT` | `0` | J20 — LIFO admission control | Opt-in only |
| `LIFO_QUEUE` | `32` | tamanho da fila LIFO se ligado | — |
| `HARDQ_DEADLINE_US` | `1500` | J24 — deadline para queries marcadas hard | Aceito |
| `HARDQ_NPROBE_EASY` | `0` | nProbe override para queries easy | Opt-in only |
| `HARDQ_DUMP_PATH` | `""` | dump de classificações para debug | Debug only |

## Runtime / hosting

| Env | Default | Notas |
|-----|--------:|---|
| `DOTNET_PROCESSOR_COUNT` | `1` | J19 — pin para 1 vCPU lógico |
| `TP_MIN_WORKERS` | `1` | J19 — ThreadPool floor |
| `TP_MAX_WORKERS` | `2` | J19 — ThreadPool teto |
| `DATASET_THP` | `1` | L3 — `madvise(MADV_HUGEPAGE)` em mmap |
| `DATASET_MLOCK` | `0` | mlock em mmap. Off por default (dispensa CAP_IPC_LOCK) |
| `FAST_JSON` | `0` | Path JSON alternativo. Sem ganho (json ~0.4µs) |
| `PROFILE_TIMING` | `0` | 1=per-request breakdown, 2=what-if instrumentation |
| `ALLOC_TRACE` | `0` | J16 — dump de allocs (zero GC confirmado) |

## Caminhos de arquivos

| Env | Default no compose | Nota |
|-----|---|---|
| `VECTORS_PATH` | `/data/references.bin` | float32 row-major, 192MB |
| `VECTORS_Q8_PATH` | `/data/references_q8.bin` | int8 quantized + scales, 48MB |
| `VECTORS_Q16_PATH` | `/data/references_q16.bin` | int16, 96MB. Default-on se presente |
| `LABELS_PATH` | `/data/labels.bin` | 1 byte/registro, 3MB |
| `IVF_PATH` | `/data/ivf.bin` | offsets + assignments |
| `IVF_CENTROIDS_PATH` | `/data/ivf_centroids.bin` | nlist × dim float32 |
| `IVF_BBOX_PATH` | `/data/ivf_bbox.bin` | bbox por cell (opcional) |
| `MCC_RISK_PATH` | `/resources/mcc_risk.json` | tabela MCC |
| `NORMALIZATION_PATH` | `/resources/normalization.json` | norma por dim |
| `CASCADE_PATH` | `/data/cascade.bin` | árvore decisão (cheat) |
| `HARDQ_MODEL_PATH` | `/data/hardq.bin` | predictor de query difícil |

## Build args (`docker build --build-arg`)

| Arg | Default | Notas |
|-----|--------:|---|
| `IVF_NLIST_BUILD` | `256` | nº de cells. Mexer requer rebuild. |
| `IVF_HEAVY_SPLIT_MAX` | `0` | cap de rows/cell (split em sub-cells). Testado=20000 → -90 pts (rejeitado) |
| `IVF_BAL_SLACK_BUILD` | `0.0` | balance constraint k-means. Não testado nesta sessão. |
