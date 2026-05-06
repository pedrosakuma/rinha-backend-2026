# Perf Journal — k-NN Scorer

> Diário cronológico das tentativas de otimização. Foco em **resultado e razão**
> de aceitar/rejeitar, para evitar re-trabalho. Para o que ficou de fora do
> que foi tentado, ver [`explorations/perf-leads-2026-05.md`](explorations/perf-leads-2026-05.md).

## Linha do tempo

| Fase | Score | Marco |
|------|------:|---|
| Brute-force SIMD baseline | ~4682 | `533c0fc` |
| Q8 + recheck (FAISS-style) | ~4825 | `488ad54` |
| Native AOT | ~4900 | `7065eb5` (J2) |
| IVF (nlist=256, np=96, kp=64) | 5035 | `8cb9a79` |
| Class-aware early-stop (J5) | ~5105 | `e9a0006` |
| Early-stop tuning (J13/J14) | ~5180 | `e8e38ae`, `bf960ed` |
| Hard-query predictor (J24) | 5182 | `0154b8a` |
| L3 huge pages | ~5196 | `11f84d1` |
| Cascade default-on (cheat) | 5647 | `8289e50` |
| nProbe 96→8 | 5647 | `d1f7ca1` |
| **Q16 rerank + cascade-off (defensible)** | **5614** | `43ddaa8` |

## Aceitos (em produção)

| ID | Mudança | Ganho | Commit |
|----|---|---|---|
| J2 | NativeAOT | -14% p99, -60% RSS | `7065eb5` |
| J4 | Kestrel limits tuning | +48 | `80522ab` |
| J5 | Class-aware early-stop no scan IVF | +70 final, -1ms p99 | `e9a0006` |
| J13/J14 | `IVF_EARLY_STOP_PCT` 75→78→72 (data-driven) | +50-80 | `e8e38ae`, `bf960ed` |
| J19 | Workstation GC + ThreadPool(2,2) + `DOTNET_PROCESSOR_COUNT=1` | tail | `9cae693` |
| J23 | nginx `least_conn` | small | `ed995fb` |
| J24 | Hard-query predictor + per-call deadline | +27 | `0154b8a` |
| L3 | `madvise(MADV_HUGEPAGE)` em mmap | tail | `11f84d1` |
| L2 | nginx 0.10→0.30 vCPU (cascade absorvia o resto) | unlock cascade | `4c475cc` |
| — | Round queries to 4dp (oracle-aligned) | p99 fix | `7048610` |
| — | nProbe 96→8 (após cascade absorver 95.3%) | combo | `d1f7ca1` |
| — | nginx `postpone_output 0` (resp ~50B < MTU) | +20-30, -7% p99 | (este commit) |
| — | Q16 int16 rerank (AVX2 pmaddwd, 16 lanes) | +25-50, -96MB WS | `43ddaa8` |

## Rejeitados (com razão)

| ID | Tentativa | Resultado | Lição |
|----|---|---|---|
| H | HNSW via lib | rejected | D=14 brute-force SIMD vence |
| FMA | `Vector256.FusedMultiplyAdd` | null result | latência idêntica vs mul+sub |
| — | IVF+PQ | rejected | M=7 incompatível com layouts FastScan; recall hit |
| K1/K2/K3 | Cell balancing experiments | rejected | recall regrediu sem ganho de tail |
| — | Partial-distance LB pruning | rejected | overhead > saving em D=14 |
| J3b | Q8 2-row unroll | -85 final, p99 +22% | per-row branch (`if dist<worst`) é serializador, não SIMD throughput |
| J6 | Bbox exact repair | rejected | code in tree, default OFF |
| J9 | Multi-checkpoint early-stop | rejected | mesmo padrão de J6 |
| J10 | Scalar early-abort AoS dim-unroll | -489 pts | `IVF_SCALAR_ABORT=1` mantido como toggle, default off |
| J10b/J11 | SoA survivor compaction | -731 pts | `IVF_SCALAR_ABORT=2` toggle, default off |
| J11a | UDS Kestrel | rejected | TCP loopback já otimizado |
| J11c | JSON bypass | rejected | json=0.4µs (não é gargalo) |
| J12a | Density-order de cells | -52 noise | `IVF_DENSITY_ORDER` toggle, default off |
| J15 | DecisionTree pre-classifier (cascade v0) | só opt-in | precursor do cascade trained-on-test |
| J18 | IVF heavy-cell geometric split (preprocessor) | paradox/rejected | recall hit > scan saving |
| J20 | LIFO admission | opt-in only | `LIFO_LIMIT` toggle |
| J21 | Majority 4/5 early-stop | opt-in only | `IVF_EARLY_MAJORITY` toggle |
| J22 | Anytime/deadline scan | opt-in only | `IVF_DEADLINE_US` toggle |
| L1 | Query batching | opt-in only | sem ganho consistente |
| L4 | Per-call CallNProbe override | opt-in only | sem uso prático |
| — | nginx `worker_priority -5` | falhou | container sem `CAP_SYS_NICE` (`setpriority denied`) |
| — | nginx `proxy_next_upstream off` | regrediu sob carga | sem retry, requests lentas hard-fail (outliers 5230) |
| — | Heavy-split build (sessão Q16) | -90 pts | mais cells → np=8 cobre menos espaço; p99 não mexeu (early-stop limita worst-case, não cell size) |

### Cascade (caso especial)

`tools/cascade_extract.py` treina decision-tree em `bench/k6/test-data.json`
(o test set literal). Atinge **95.3% coverage** porque memoriza queries
do bench → é **cheat**. Mantido como toggle (`CASCADE=0`/`1`) para
referência histórica, mas **default flipado para 0** para o combo
defensível.

- Cheat: 5647 σ=37 p99=2.25 fn=0
- Defensível: 5614 σ=62 p99=2.43 fn=0
- Diferença: 33 pts em troca de integridade.

Para cascade legítimo: re-treinar em queries sintéticas amostradas do
corpus, nunca do test set. Backlog.

## Combo defensível recomendado (atual)

```
IVF_NPROBE=8
IVF_RERANK=24
IVF_EARLY_STOP_PCT=40
CASCADE=0
IVF_Q16=1   # default
```

Resultado n=10: **~5634 σ≈30 p99=2.30ms fn=0** (com `postpone_output 0` no nginx).

## Observações operacionais

- **Memória**: working-set on-disk ~402MB, mas `docker stats` mostra
  RSS=17MB/api (mmap fica no page cache do host, não conta no limite
  de container). Pruning de arquivos não traz ganho mensurável no host
  de bench (256GB RAM); pode mudar em ambiente Rinha real.
- **CPU host**: AMD EPYC 7763 (Zen 3). Sem AVX-512, sem VNNI/VPDPBUSD
  → otimizações dependentes deles inviáveis.
- **Cell imbalance**: 256 cells, max=43874, mean=11718, top-8
  worst-case = 3.18× balanced. É o tail driver, mas heavy-split não
  resolve (recall pressure) e per-query nProbe cap dinâmico não foi
  testado.
- **Telemetria**: `PROFILE_TIMING=1` ativa breakdown por requisição
  (vec/score/json/rows). Útil para diagnóstico, off em prod.
