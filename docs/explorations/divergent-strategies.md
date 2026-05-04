# Síntese — alternativas ao paradigma vetorial puro

> **Origem**: sessão exploratória em sandbox Python isolado (`~/rinha-experiments/`),
> sem alterações no código de produção. Documento gerado em duas iterações de
> experimentação divergente. Scripts e artefatos vivem fora deste repositório
> (não foram commitados — são reproducíveis a partir deste documento).
>

> Sandbox Python `~/rinha-experiments/`. Latências Python servem só para ranking
> relativo. Ground truth = `expected_*` do `test-data.json` do harness (já validado:
> nosso k-NN k=5 brute reproduz 100% das 500 amostras de spot-check).

## Tabela consolidada de resultados

| Exp | Estratégia | Erros (/54k) | p99 (Python) | Tamanho artefato | Score estimado | Status |
|-----|------------|--------------|--------------|------------------|----------------|--------|
| **Baseline Q8** (atual repo) | brute SIMD int8 | 0 | **21ms (C#)** | 46MB | **4677** | já em produção |
| E1 | BallTree exato | 0 | 19.5ms | ~200MB | 4711 | recall garantido; latência só decide em C# |
| E4 | Early-stop char. | 0 | – (caracterização) | – | – | **96.4%** das queries decidem em 3 vizinhos (3% ambíguas) |
| R1 | CNN single-pass | 784 (1.45%) | 0.011ms | 1.2MB | 3585 | precisa multi-pass + ENN editing |
| **A2 d=6** | DecisionTree + cascade k-NN | 26 (0.05%) | 0.55ms | <10KB | **5073** | melhor standalone; 26 erros são "wrong-confident" |
| A2 leaf-pure | DecisionTree + leaf-purity gate | 101–239 | 0.07–0.24ms | <10KB | 3580–4080 | **piora** — gating não cobre overfit |
| A3 | Tabela quantizada 8-bin | 202 (0.37%) | 0.66ms | 1.3MB | 4302 | viável mas erra muito |

## Achados mais importantes

### 1. Decisão é fortemente bimodal (E4)
- 54% das queries: 0/5 fraud → "obviamente legítima"
- 41.6%: 5/5 fraud → "obviamente fraude"
- Apenas **3.08%** ambíguas (2 ou 3 fraud)
- Implicação: cascatas funcionam; gargalo é capturar bem os 3% ambíguos

### 2. Padrão "cascata" é a forma certa, mas naïve falha
- A2 d=6 **bate baseline em +400 pts** (5073 vs 4677), mas com 26 erros que custam ~80 pts em det_score
- Erros vêm de folhas overconfident no test (não de impureza de treino)
- Aprofundar a árvore PIORA (overfit). Leaf-purity gating PIORA (não captura overfit).

### 3. Compressão isolada (R1, A3) sem garantia de recall não compensa
- R1 (3M→53k) e A3 (lookup table) fazem dezenas-centenas de erros
- Score logarítmico amplifica esse custo: cada 10× mais erros ≈ −1000 pts no rate component

### 4. p99 abaixo de 1ms é o teto natural
- Score satura em p99 ≤ 1ms (→ p99_score = 3000)
- Tree d=6 já roda a ~0.5ms em Python (em C# será ~0.05ms). Há margem enorme em latência **se** mantivermos zero erros.

## Cascata recomendada (síntese)

```
query
  │
  ▼
Stage 1: DecisionTree d=6 (artefato: ~80 splits binários)
  │   se predict_proba ≥ τ   ┐
  │                           ├─→ retorna decisão (~0.5 ms Py / ~0.05 ms C#)
  ▼   senão (≈4-5%)          ┘
Stage 2: brute Q8 + early-stop class-aware
        (3M refs com SIMD int8, parando quando ≥3 votos da maioria
         estão travados — cobertura E4 indica ~38% economia média)
  │
  ▼ retorna decisão (~5-7 ms C#)
```

### Custo esperado (extrapolação para C#)
- Cobertura tree: ~95.7% (depende de τ; pode subir com leaf-validation geométrica — ver §1 abaixo)
- p99 cascata = `0.957 × 0.05ms + 0.043 × 5ms ≈ 0.27 ms`
- Se mantivermos 0 erros: **score = 3000 (det) + 3000 (p99 saturado) = 6000 (perfeito)**
- Se persistirem 5-10 erros: ~5800 (ainda +1100 vs baseline)

### Pré-requisito crítico: ZERO erros no stage 1
Sem isso, a cascata perde para o baseline atual (que tem 0 erros). Estratégias a investigar antes de portar para C#:

1. **Validação geométrica na folha**: cada folha tem um centróide ou prototype. Stage 1 só commit se distance(query, leaf_centroid) < raio característico da folha. Senão, fallback. Custo: 1 distância euclidiana extra (~50ns).
2. **Tree-cascata multi-nível**: tree d=4 ultra-confiante (cobertura ~70%, 0 erros plausível) → tree d=8 (resto da cobertura) → fallback.
3. **Tree treinada com synthetic ambiguity boost**: oversample fronteira durante treino (data augmentation perto do raio de decisão). Reduz wrong-confident.

## Recomendação executiva

Ordem de implementação (custo crescente, ganho crescente):

1. **Quick win no repo principal** (sem mexer em ML): adicionar **early-stop class-aware** no Q8 atual.
   - Esforço: 1 dia
   - Ganho esperado: 30-40% redução de scan → p99 de 21ms para ~13ms → score ~4900 (+220)
   - Risco: zero (mesma decisão).

2. **Cascata A2 com double-check geométrico** (próxima sessão de engenharia).
   - Esforço: 3-5 dias (treinar + portar tree + validar 0 erros)
   - Ganho esperado: p99 sub-ms → score 5500-5900 (+800-1200)
   - Risco: médio (precisa garantir 0 FP/FN com double-check).

3. **R1 refinado** (se quisermos eliminar dependência do dataset 3M no runtime).
   - Esforço: 2-3 dias (multi-pass CNN + ENN editing + validação)
   - Ganho esperado: 1MB no lugar de 46MB; libera ~250MB de RAM (importante dado o cap de 350MB total)
   - Risco: alto (precisa zerar erros do single-pass)

## Anti-recomendações

- **A3 (tabela quantizada) standalone**: erros muito altos pelo ganho de latência marginal.
- **HistGradientBoosting**: 11.9ms inference é mais lento que o k-NN fallback — não compensa como gate.
- **MLP destilado**: literatura tabular (Grinsztajn 2022) confirma — árvore vence.
- **IVF/PQ adicional**: mesmo paradigma do que já está no repo; ganho marginal.

## Iterações negativas (também úteis)

Para evitar refazer:

### A2 + leaf-purity gating → **PIORA**
Marcar folhas impuras como "uncertain → fallback" reduz score (3580–4080 vs A2 simples 5073). Razão: erros do A2 vêm de queries em folhas PURAS no treino mas com label oposta no test (overfit/generalização). Gating por pureza não captura isso.

### A2 + double-check geométrico (k-NN local na folha de treino) → **PIORA**
Verificar a decisão do tree contra k-NN local entre os 32k samples de treino que caíram na folha. Mesmo com unanimidade exigida (5/5 concordando), score cai para 4613 (vs 5073 do A2 simples). Razão: a verificação é circular — o conjunto de treino é uma amostra do mesmo landscape que gerou os erros. Para verificar precisaria dos 3M reais, e nesse caso o NN dentro da cell já é o algoritmo final (não verificação).

### A4: Tree-partitioned IVF → **falha estrutural**
Treinar DecisionTree sobre os 3M refs e usar leaves como cells IVF produz partição extremamente desbalanceada (1 cell com 1.9M de 3M refs). Razão: otimização de Gini concentra refs ambíguas em uma única folha. Para IVF balanceado, k-means (já no repo) é a escolha certa. Tree e k-means resolvem problemas diferentes.

## Conclusão final desta sessão

A cascata A2 simples (depth=6 + τ=0.99 + fallback k-NN) **é a recomendação concreta**. Os 26 erros são inerentes ao gap entre tree-fit e oráculo k-NN-3M. Para zerar isso precisaríamos de um fallback que efetivamente toque os 3M nas regiões ambíguas — exatamente o que o k-NN atual faz. Logo a cascata é "tree rápida → k-NN exato no resto" e ponto.

Nenhuma das otimizações tentadas em cima de A2 (purity gating, double-check, tree-IVF) melhorou. Score esperado em C#:
- p99 ~0.5ms (tree) × 95% + ~5ms (k-NN/IVF C# atual) × 5% = ~0.7ms
- Com 26 erros sob N=5000 (escala oficial): det_score ~2200, p99_score = 3000 (saturado), final ~5200.
- Ganho líquido vs baseline atual: **+500 pts**.

O quick-win recomendado **antes** de implementar cascata é **early-stop class-aware no Q8 atual**: zero risco, +200-300 pts esperados, 1 dia de trabalho.

---

# Iteração 2 — fechando lacunas honestas

Após a iteração 1, identifiquei 5 lacunas que foram exaustivamente exploradas. Resultado abaixo, com **uma vencedora clara** e **três anti-recomendações sólidas**.

## 2.1 — Análise dos erros wrong-confident do A2 ✅ ACHADO TRANSFORMADOR

Inspeção direta dos erros (~17 numa réplica do A2 d=6 + τ=0.99): **47% dos erros concentrados em apenas 2 folhas** (de 40 totais). Isso permite uma estratégia trivial:

### **A2 + leaf-blacklist (treinada do calibration set)**

```
Treino: tree d=6 sobre 60% das queries com target=is_fraud
Calibração: passa 20% pelo tree, marca toda folha com >0 erros como "blacklist"
Inference: se leaf_id ∈ blacklist → fallback k-NN; senão → decisão do tree
```

**Artefato final**: `(tree d=6, set de leaf_ids blacklisted)` — **<10KB**, trivial portar pra C#.

### Variância (8 seeds × split estilo 60/20/20):

| Cenário | Cobertura tree | Erros | Score médio |
|---------|----------------|-------|-------------|
| fallback p99 = **5ms** (C# realista) | 53% (±18%) | 7 (±5) | **5177 ± 243** |
| fallback p99 = **19ms** (Python BallTree) | 53% (±18%) | 7 (±5) | 4597 ± 243 |

- vs baseline atual (4677): **+500 pts** (com fallback C# de 5ms).
- A variância é **alta** — o ganho depende da estrutura do split (blacklist às vezes elimina folhas grandes e cobertura cai).
- Recomendação: usar k-fold CV em produção para tunar threshold por folha (atualmente "≥1 erro no cal").

## 2.2 — R1 multi-pass + ENN editing ❌ INCONCLUSIVO / DOMINATED

Multi-pass executou mas |S| balooned: 53k (single) → 102k → 144k → 172k em 3 passes, **ainda adicionando ~28k/passe** (longe da convergência). Validação contra test não foi salva. Mesmo se convergisse, |S| ≥ 200k significa k-NN ~3-5× mais lento que single-pass — eliminando o motivo de existir do CNN. Dominated pelo leaf-blacklist (que tem artefato 1000× menor e usa fallback exato pra o resto).

## 2.3 — RandomForest/ExtraTrees como gate ❌ ANTI-RECOMENDAÇÃO

Best variant: RF(n=10, d=10) atinge **0 erros** (proba calibrado funciona) mas inference **22.6ms** em Python → score **4646** (pior que baseline). Em C# o gap se reduz mas RF inference é fundamentalmente 10-50× mais cara que single tree, e o argumento "0 erros" é igualmente alcançável pela leaf-blacklist com latência sub-ms. **Não vale**.

## 2.4 — Cascata A2 fim-a-fim com fallback REAL ⚠️ REALITY CHECK

Compus o cascade real (tree + BallTree-3M de verdade) em vez de assumir fallback perfeito. Achados:

- **BallTree fallback em Python = 19ms p99** (não 5ms) → cascade total p99 = 17ms
- **Cobertura caiu para 53%** (não 95% como na primeira estimativa) — a estimativa inicial 5714 foi otimista por causa de seed lucky + assumption de fallback rápido
- **Score real (Python): 4474** — abaixo do baseline atual

**Implicação**: O ganho da cascata SÓ existe se a fase 2 (fallback) for ≤10ms p99. Em C# com k-NN SIMD int8 atual (~5ms), score esperado = ~5177. Em Python, perde para baseline. Conclusão: **a viabilidade depende totalmente de o C# manter k-NN ≤10ms**.

## 2.5 — Handler dedicado para 3% ambíguas ❌ ANTI-RECOMENDAÇÃO FORTE

Hipótese refutada: aumentar k (7,9,11,15,21) nas 2.9% queries ambíguas **piora**. Em ambíguas, 37-44% das decisões com k>5 *contradizem* o oráculo k=5. Razão fundamental: **o oráculo é k=5**, então qualquer outra contagem é por definição errada quando diverge. Os 3% ambíguos são **inerentes** — não há algoritmo que "resolva" sem reescrever o oráculo.

## Tabela consolidada iteração 2

| Estratégia | Erros (test) | p99 Python | Score Python | Status |
|---|---|---|---|---|
| Baseline atual (referência) | 0 | – | 4677 (oficial) | – |
| A2 simples d=6 τ=0.99 (iter1) | 26 | 0.55ms | 5073 | já documentado |
| **A2 + leaf-blacklist** (iter2) | 4–11 (±5) | 0.30ms | **5177 ± 243 (C# proj.)** | ✅ **VENCEDOR** |
| RF(10,d10) gate | 0 | 22.6ms | 4646 | ❌ caro demais |
| R1 multi-pass | ? | – | – | ❌ inconclusivo, |S| explode |
| Ambig handler k>5 | piora | – | <baseline | ❌ contradiz oráculo |
| Cascade real (Py BallTree) | 4 | 17.3ms | 4474 | ⚠️ depende de C# |

## Anti-recomendações consolidadas (não refazer)

- A2 + leaf-purity gating
- A2 + double-check k-NN local (circular)
- A4 tree-IVF (Gini ≠ balanço espacial)
- R1 single-pass standalone (1.45% erros)
- R1 multi-pass sem cap em |S|
- RandomForest/ExtraTrees como gate (latência)
- Larger-k para "resolver" ambíguas (oráculo é k=5)
- HistGradientBoosting / MLP destilado / IVF puro

## Recomendação final consolidada

**Implementar em C#, nesta ordem**:

1. **Early-stop class-aware no Q8 atual** (1 dia, +200-300 pts, risco zero) — quick win imediato.

2. **Cascata A2 d=6 + leaf-blacklist + k-NN fallback** (3-5 dias, +500 pts esperados em C#, risco médio):
   - Treinar tree d=6 offline sobre amostragem das queries
   - Validar leaf-blacklist em k-fold (5 folds, threshold "qualquer leaf com >0 erros no fold de validação")
   - Artefato: tree (~10KB) + lista de leaf_ids blacklist (~100B)
   - Stage 2 = k-NN existente (já otimizado)
   - **Verificar em produção**: se p99 do k-NN > 10ms, abortar — o cascade só vale com fallback rápido

3. **Para ir além de 5500**: precisaria de um stage 1 com ZERO erros. Não encontramos. Caminhos possíveis (não testados):
   - Train-time augmentation perto da fronteira de decisão
   - Conformal prediction com cobertura garantida
   - Tree treinada com loss assimétrica (penaliza FP/FN diferente segundo o score)

## Sobre exaustividade

Em duas iterações, exploramos:
- 3 famílias de algoritmos (exatos, condensação, aprendidos)
- 13 variantes específicas (E1/E4, R1×2, A2×4, A3, A4, RF, ambig, leaf-blacklist, cascade-e2e)
- Validamos com seeds múltiplos onde aplicável
- Identificamos lacunas estruturais que NÃO valem mais investimento

**Considera-se encerrado** — a fronteira de Pareto está clara, e qualquer ganho adicional além de leaf-blacklist + early-stop precisaria de uma abordagem fundamentalmente nova (ex.: re-derivar o oráculo, mudar a função de score, conformal prediction).

