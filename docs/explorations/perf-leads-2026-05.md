# Leads de Performance — k-NN Scorer (maio/2026)

> **Origem:** deep research técnico sobre estado-da-arte de k-NN/ANN e otimizações
> aplicáveis ao cenário concreto do projeto. Relatório completo em
> `~/.copilot/session-state/<id>/files/knn-perf-research.md` (artefato fora do repo).
>
> **Cenário:** D=14, N=3M, k=5, L2², recall=100% obrigatório,
> 0.45 vCPU × 2 réplicas, 350MB RAM total, AVX2 (sem AVX-512), .NET 10.
>
> **Diagnóstico central:** `IvfScorer` com Q8 + early-stop class-aware está dentro
> de **1.5–4×** do piso teórico. Não existe algoritmo publicado em 2023–2025 que
> bata essa abordagem para esse cenário específico **mantendo recall=100%**. O gap
> residual é dominado por **CFS jitter** (5–10ms), não pelo scorer.

## Leitura rápida — o que descartar e o que perseguir

### ❌ Descartar (não aplicáveis a D=14 / recall=100% / AVX2)

| Tecnologia | Motivo da rejeição |
|---|---|
| HNSW / DiskANN / KD-tree / Ball-tree / VP-tree | Em D=14 com recall=100% ficam **mais lentos** que brute-force SIMD. KD-tree visita ~N^(13/14) ≈ 2.1M nós; HNSW degrada por falta de "atalhos" e não garante recall=100% sem efK=N. |
| RaBitQ / Extended-RaBitQ (SIGMOD 2024/2025) | σ_erro ∝ 1/√D. Para D=14, K' precisaria ser ~30× → líquido **3.7× mais lento** que Q8. Papers explicitamente para D≥96. |
| 4-bit PQ FastScan (FAISS) | Exige M múltiplo de 16; nosso M=7 com dsub=2 quebra o layout VPSHUFB. |
| LVQ (Intel SVS) | Requer AVX-512 para performance otimizada. Hardware-alvo é AVX2. |
| Anisotropic VQ (Google ScaNN) | Para inner product e D≥128. Sem ganho para L2 em D=14. |
| Conformal prediction como gate | Com α=0 degenera para `purity=1.0` (já implementado). Com α>0 viola recall=100%. |
| AVX-512 / AVX-VNNI / Vector512 | Sem hardware. Vector512 em AVX2 emula em software (mais lento que 2× Vector256). |
| io_uring no Kestrel | Workload é CPU-bound, não I/O-bound. Sem ganho. |
| Learned indexes (ELPIS, LEAP) | Para D=14, k-means do IVF já é "learned" o suficiente. Ganhos só para recall<100%. |
| LAESA / triangle-inequality bounds | Combinado com IVF+Q8 já existente, lookup de pivôs custa mais que o próprio Q8 scan. |

### ✅ Perseguir (priorizado por ganho × esforço)

Detalhes em cada lead abaixo.

| # | Lead | Ganho p99 | Esforço | Risco |
|---|---|---|---|---|
| L1 | Query batching (1 scan ↦ N=4–8 queries) | **−30 a −50%** | Alto | Médio |
| L2 | Cascata expandida com validação geométrica de folha | **−20 a −40% p50/p90** | Médio | Médio |
| L3 | THP (`MADV_HUGEPAGE`) nos mmaps | **−5 a −15%** | Baixo | Baixo |
| L4 | Adaptive nProbe (LUT por query difficulty) | −5 a −10% | Baixo | Baixo |
| L5 | NativeAOT | −3 a −8% jitter | Médio | Baixo |
| L6 | OPQ (rotação ortogonal antes de PQ) — só para `IvfPqScorer` | −5 a −10% no PQ scorer | Médio | Baixo |
| L7 | Mitigação de CFS jitter (cpuset/quota tuning) | **dominante no p99** | Baixo | Médio (host) |
| L8 | Branchless CMOV `InsertTopK` | <1% | Baixo | Baixo |

---

## L1 — Query batching (N=4–8 queries por scan)

**Ideia.** Em vez de 1 query → 1 scan de 562k vetores Q8 (NPROBE=96), agrupar
N=4–8 queries chegando dentro de uma janela curta e fazer **um único scan**
calculando N distâncias por linha. A linha do dataset é lida uma vez do L1;
amortiza-se memory bandwidth entre N queries.

**Por que é a maior alavanca.** Análise do relatório:
- Piso de bandwidth p/ scan IVF+Q8 1-query: ~2.2 ms
- Piso de bandwidth p/ scan IVF+Q8 com N=4: **~0.55 ms**
- Bandwidth efetivo de 0.45 vCPU compartilhando LLC: 4–8 GB/s (frações do socket)

**Esboço técnico.**
- Buffer LIFO atual (`Program.cs:111-133`) coleta requests pendentes.
- Worker thread agrupa até N queries (deadline-bounded, ex.: 500µs).
- Cada query é convertida para Q8 + IVF probe set (top-NPROBE) **independente**.
- Scan agrupado: para cada cell c na união dos probe sets de todas N queries,
  scan as linhas Q8 acumulando N distâncias por linha (em N registradores).
- Top-K' por query mantido em buffers separados.
- Rerank float + decisão de fraude por query.

**Complicações.**
- Probe sets divergem entre queries → união pode ter mais cells que NPROBE de
  uma query sozinha (mas ainda menos que N×NPROBE). Para queries similares, a
  união é compacta.
- Q8 row layout AoS: ler 1 linha (16B) e fazer N VPMADDWD com N qWide
  pré-carregados é trivial (cabe em registradores: q_widened ocupa 1 YMM cada,
  N=4 usa 4 YMM, sobra para r_widened + diff + acc).
- Early-stop class-aware fica mais complexo (cada query tem seu checkpoint).
  Sugestão: aplicar early-stop por query individualmente, mas continuar
  scan para as outras queries no batch (a economia vem da amortização de B/W).
- Latência de espera do batch < 500µs para não machucar p50 sob baixa carga.

**Critério de aceitação.** Throughput de queries/s aumenta ≥30% mantendo
p99 ≤ p99 atual. Recall=100% no harness oficial.

**Risco.** Refator significativo do pipeline request → scorer → response.
Pode exigir mover a Q8 quantization para fora do scorer.

---

## L2 — Cascata expandida com validação geométrica de folha

**Estado atual.** `Cascade.cs` é DT depth=4 com 1 folha pura (leaf 2: n=28795,
purity=1.0, pred=legit). Cobre ~27% das queries, todas legítimas óbvias.

**Targets de cobertura (do `divergent-strategies.md`).**
- 54% das queries: 0/5 fraud
- 41.6%: 5/5 fraud
- 3.08% ambíguas
- DT d=6 puro hipotético: cobertura ~95.7% mas 26 erros (inaceitável)

**Proposta.** Adicionar **validação geométrica por folha** para garantir
zero erros ao expandir cobertura:

1. Para cada folha pura, pré-computar:
   - `centroid_leaf` = média dos pontos de treino na folha (14-d)
   - `radius_leaf` = max(dist(p, centroid_leaf)) sobre os pontos de treino
2. Em runtime, commit a folha apenas se:
   - `dist²(query, centroid_leaf) < (radius_leaf × safety)²` (ex. safety=0.9)
   - **E** opcionalmente `top3Dists[0] < threshold_per_leaf` (já temos top-3 IVF)
3. Custo extra por query: 1 distância 14-d (~50ns) + 1 comparação.
4. Fallback: `IvfScorer` normal.

**Cobertura esperada.** 27% → 50–75% (depende de `safety`). Ganho real é em
**p50/p90** (queries no fast-path saem em ~50µs em vez de 3–8ms), o que
desafoga CPU para as ambíguas, melhorando p99 indiretamente.

**Sub-leads.**
- **L2a:** Treinar DT mais profunda (d=6, `min_samples_leaf=100`,
  `purity_threshold=1.0`) e verificar se há folhas puras adicionais
  significativas. Tooling existe em `tools/cascade_emit.py` + `cascade/tree.json`.
- **L2b:** Cascata multi-nível (DT d=4 ultra-confiante → DT d=8 →
  fallback IVF), conforme proposta no `divergent-strategies.md` §1.
- **L2c:** Tree treinada com synthetic ambiguity boost (oversample de
  fronteira de decisão) para reduzir wrong-confident.

**Critério de aceitação.** Zero erros adicionais no harness oficial;
cobertura medida do fast-path ≥50%.

**Risco.** Erro do classificador é amplificado pelo scoring logarítmico
(cada 10× mais erros ≈ −1000 pts). **Pré-requisito:** validação geométrica
provada experimentalmente em sandbox antes de merge.

---

## L3 — THP / hugepages para os mmaps

**Diagnóstico.** Dataset Q8 = 48 MB; float = 192 MB. Em pages de 4KB,
isso é 12k–48k entradas TLB. L1 dTLB tem 64 entradas, L2 sTLB ~1500.
Random access nesse footprint causa TLB misses constantes.

**Solução.** Após `MemoryMappedFile`+`MemoryMappedViewAccessor`, chamar
`madvise(ptr, size, MADV_HUGEPAGE)` (P/Invoke). Kernel promove pages 4KB
em hugepages 2MB. Footprint vira 24–96 entradas — cabe inteiro no L1 dTLB.

**Pré-requisito do host.**
```bash
cat /sys/kernel/mm/transparent_hugepage/enabled
# precisa ser [always] ou [madvise]
```

**Implementação esboço.**
```csharp
// after acquiring SafeBuffer pointer in Dataset.cs
[DllImport("libc", SetLastError = true)]
static extern int madvise(IntPtr addr, UIntPtr length, int advice);
const int MADV_HUGEPAGE = 14;
madvise((IntPtr)_vectorsPtr, (UIntPtr)totalBytes, MADV_HUGEPAGE);
```

**Complicações.**
- mmap precisa estar alinhado a 2MB. Se não estiver, kernel só promove o
  miolo. `MemoryMappedFile` não dá controle direto; pode requerer
  `mmap` via P/Invoke com `MAP_HUGETLB | MAP_ANONYMOUS` + `read()` do
  arquivo, ou pre-alocação de buffer alinhado.
- Container precisa ter THP habilitado (não isolado por cgroup).
- Verificar via `grep AnonHugePages /proc/<pid>/smaps` que as pages foram
  efetivamente promovidas.

**Critério de aceitação.** `/proc/<pid>/smaps` mostra AnonHugePages ≥ 90%
do dataset. Bench mostra ≥5% redução no scan time.

**Risco.** Baixíssimo: se THP não disponível, `madvise` retorna sem efeito.

---

## L4 — Adaptive nProbe por query difficulty

**Ideia.** Hoje `IVF_NPROBE` é constante (default 96). Mas:
- Queries cujo top-1 centroid está **muito próximo** (low `cellsDist[0]`):
  provavelmente "fácil", nProbe baixo (ex. 32) é suficiente.
- Queries cuja distância aos top centroids é **alta e plana**:
  query "rara/fronteira", precisa nProbe alto (ex. 128).

**Implementação.** LUT/heurística em runtime:
```csharp
int adaptiveNProbe = ComputeAdaptive(cellsDist[0], cellsDist[3]);
// e.g.: if cellsDist[0] < τ_low → 32; if ratio cellsDist[3]/cellsDist[0] > τ_flat → 128;
// else 96.
```

**Validação.** Telemetria já existe (`LastEarlyStopMode`, `LastRowsScanned`).
Usar bench/k6 + recall-check.sh para validar zero perda de recall em diferentes
nProbe targets.

**Ganho esperado.** −5 a −10% no scan médio sem perder recall (queries fáceis
processadas mais rapido).

**Risco.** Definir threshold sem treino formal; pode exigir calibração em
dataset offline.

---

## L5 — NativeAOT

**Análise honesta.** Para hot loops SIMD já em tier-1, AOT vs JIT são
**idênticos** (ambos emitem AVX2). Ganho real: eliminação de jitter de tier
transitions, footprint −50MB (sem JIT runtime), startup mais rápido.

**Onde dói:** sob saturação CPU, spikes de tier-0→tier-1 contribuem para p99
outliers. AOT remove essa fonte.

**Esforço.** Médio. STJ source-gen já está em uso; é a maior dependência
AOT-incompatível removida. Riscos:
- Validar se `MemoryMappedFile` funciona em AOT (sim, em .NET 8+).
- `Environment.GetEnvironmentVariable` lookups frequentes podem ser
  problema em startup mas não em hot path.
- Trim warnings em `Microsoft.AspNetCore.*`.

**Critério.** p99 reduzido em ≥3% sob carga sustentada de 30s.

---

## L6 — OPQ (rotação ortogonal) para `IvfPqScorer`

**Aplicabilidade.** Só faz sentido se `IvfPqScorer` for ativado em produção
(hoje a configuração padrão usa `IvfScorer` Q8). Se PQ for caminho secundário,
deprioritizar.

**Ganho técnico.** OPQ aprende rotação O ∈ ℝ^{14×14} que minimiza distorção
de PQ. Para D=14, M=7, dsub=2, OPQ pode reduzir erro de quantização em 10–20%
→ K' menor para mesma recall@K → −5 a −10% no tempo total do `IvfPqScorer`.

**Esforço.** Treino com sklearn/numpy no Preprocessor (1–2 dias). Aplicação
da rotação na query: 14×14 matmul = ~196 flops, desprezível.

---

## L7 — Mitigação de CFS jitter (orçamento de host/compose)

**Análise crítica.** Com `--cpus=0.45`, CFS define
`cpu_quota=45000us, cpu_period=100000us`. Pior caso: processo bloqueado por
55ms a cada 100ms. **Esse é o maior contribuinte para p99 hoje** (relatório
estima 5–10ms de jitter).

**Opções (todas mexem em `docker-compose.yml`).**
1. **`--cpu-period=10000` + `--cpu-quota=4500`**: reduz granularidade
   (10ms de janela em vez de 100ms). Jitter p99 cai proporcionalmente.
2. **`--cpuset-cpus=0,1` por réplica + repartir quota**: pin CPU para
   reduzir migração de cores e cache pollution.
3. **Redistribuir budget de CPU entre lb (0.10) / api1 (0.45) / api2 (0.45)**:
   se nginx está sub-utilizado, mover budget para apis.
4. **Sysctl no host** (`kernel.sched_min_granularity_ns=1000000`) — mas exige
   acesso privilegiado.

**Critério.** Reduzir jitter mensurado via `/sys/fs/cgroup/cpu.stat`
(`nr_throttled`, `throttled_time`).

**Risco.** Mudanças no compose podem violar regras da rinha de backend
(verificar regulamento sobre soma de CPU).

---

## L8 — Branchless CMOV `InsertTopK`

**Análise.** Guard `if (dist < worst)` filtra ~99.999% das chamadas no scan
brute-force. Misprediction custa ~15 ciclos × ~5/3M chamadas = desprezível.
Ganho previsto **<1%**.

**Veredicto.** Não vale o esforço. Listado por completude.

---

## Apêndice: limites teóricos calculados

| Cenário | Tempo de scan |
|---|---|
| Brute-force Q8 puro (3M × 16B) | ~5–7ms (compute) / ~8ms (B/W) |
| IVF+Q8 NPROBE=96 (562k vetores) | ~1.7ms scan + 32KB centroid scan |
| IVF+Q8 + early-stop 75% (típico) | ~1.3ms |
| IVF+Q8 + batch N=4 (amortizado) | **~0.35ms / query** |
| **Piso de bandwidth absoluto (N=4 batch)** | **~0.55ms / query** |

Overhead estimado de framework (parse JSON, route, marshal): ~0.3–0.5ms.
Gap atual entre p50 (~2–4ms) e o piso teórico (~1ms): margem realista.
Gap entre p99 (~15–21ms) e p50: dominado por CFS jitter, não algoritmo.

## Apêndice: papers de referência

1. **RaBitQ** (SIGMOD 2024): arXiv:2405.12497
2. **Extended-RaBitQ** (SIGMOD 2025): arXiv:2409.09913
3. **FAISS Library** (atualizado 2026): arXiv:2401.08281
4. **Anisotropic VQ / ScaNN** (NeurIPS 2020): arXiv:1908.10396
5. **SOAR** (NeurIPS 2023): arXiv:2404.00774
6. **Intel SVS / LVQ**: VLDB 2023, github.com/intel/ScalableVectorSearch
7. **.NET 10 Performance**: devblogs.microsoft.com/dotnet/performance-improvements-in-net-10/
8. **ANN-Benchmarks**: ann-benchmarks.com (datasets mín. D=25 — não cobrem nosso D=14)

## Como consumir em outra sessão

Sugestão de prompt inicial para a próxima sessão:
> Leia `docs/explorations/perf-leads-2026-05.md`. Quero implementar o
> lead **L<N>**. Comece propondo o desenho detalhado e os pontos de
> mudança no código antes de mexer em qualquer arquivo.

Para validar qualquer alteração, usar:
- `bench/run.sh` (smoke local)
- `bench/recall-check.sh` (garantir recall=100%)
- `bench/run-x3.sh` ou `run-xN.sh` (estabilidade do p99)
