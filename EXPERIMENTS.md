# Log de Experimentos

Registro das rodadas de treino do projeto (detector + classificador), pra reconstruir o histórico depois (útil pro relatório de IC). Este log **começa em 2026-07-16**, quando foi criado — só tem entradas verificadas com confiança nesta sessão. Rodadas anteriores não entraram aqui pra não arriscar registrar hiperparâmetro/número de memória de forma imprecisa; ver `weights/backups/manifest.md` para o que se sabe com certeza sobre os checkpoints antigos preservados (só data e contexto factual, sem detalhe de treino não confirmado).

Convenção: preencher uma linha nova a cada rodada de treino completa (não a cada ajuste de config testado localmente).

## Classificador

| Data | Checkpoint | Dado de treino | Resultado | Notas |
|---|---|---|---|---|
| 2026-07-12 | `weights/classifier_best.pt` (ativo) | 100% sintético — N1 200 amostras/classe (1232 classes), OUTROS 20000 treino / 5000 val, sem nenhum dado real misturado (`CLASSIFIER_SAMPLES_TRAIN=200`, `CLF_OUTROS_SAMPLES_TRAIN=20000`) | Recall no benchmark de região (Manga109, oficial `<text>`): **82.6%** (1 página) / **64.0%** (4 páginas curadas manualmente) / **78.0%** (corpus completo, 8536 páginas com N1, 109 volumes, 42611 pares esperados) | Resolveu o colapso de atalho de domínio (real vs. sintético) de rodadas anteriores ao remover dado real do treino inteiramente. **Ainda carrega o problema de erosão** (`CLF_MORFO_K_MAX=3` na época) corrigido só depois deste treino (ver abaixo) — não foi retreinado com a correção ainda. |

### Pendências conhecidas não incorporadas neste checkpoint
- `CLF_MORFO_K_MAX` corrigido de 3→2 em 2026-07-16 (auditoria de filtros mostrou penhasco de acurácia 92.5%→40% entre k=2 e k=3) — **não retreinado ainda**.
- WARN residual de legibilidade em `generate_sample` (~12.5% num lote de 200 sanity, medido em 2026-07-19 com o código atual — não é "tofu box"/fonte sem glifo, já descartado antes: é contraste/fill_ratio fora da faixa mesmo após todos os retries). Achado nesta medição: 14px estava 2.5x sobre-representado nos WARN vs. sua taxa de amostragem (40% dos WARN, 16% da amostragem) — peso de `CLASSIFIER_FONT_SIZE_WEIGHTS` pra 14px reduzido de 16→6 (redistribuído pra 16/18px). Efeito medido: **25→21 WARN em 200 amostras (12.5%→10.5%)** — melhoria modesta, não elimina o problema. Investigação mais funda mostrou que o residual é dominado por combinação **fonte fina (Klee-One, Yuji-Boku, Hachi-Maru-Pop) × kanji de muitos traços**, não por tamanho — corrigir isso de verdade exigiria uma lista tipo `CLF_FONTES_PESADAS` só que pro sentido inverso (fontes finas, forçar só "dilate"), não feito ainda por decisão de priorizar o retreino dado o prazo apertado (1 mês) da IC. **Não bloqueia o retreino** — a função sempre emite uma amostra usável, o WARN só sinaliza qualidade sub-ótima numa fração pequena.

## Detector

| Data | Checkpoint | Dado de treino | Resultado | Notas |
|---|---|---|---|---|
| 2026-07-18 (rodada corrigida) | `weights/best.pt` (ativo) | ~17 imagens reais (Roboflow) + 300 páginas sintéticas (`generate_pages.py`, grade 2D multi-coluna em `dividir_em_celulas`), confirmadamente fundidas desta vez (~38 batches/epoch, batch=8 -> ~300 imgs/epoch, condizente com o dataset completo). `val` 100% real. 150 epochs, `yolo26n.pt`, imgsz 1024, batch 8. | **mAP@50 0.81** (epoch 146 pico 0.813, epoch 150 final 0.810), mAP@50-95 ~0.51, precision ~0.90, recall ~0.78-0.79 | Corrigiu o bug de resolução de diretório do Manga109 no Kaggle (`_buscar_manga109_diretorio` escolhia o dataset Roboflow por engano — ver `config.py`/`generate_pages.py`) que fez a rodada anterior (abaixo) rodar com bem menos dado do que deveria. `detector_fp_check.py` (150 páginas sem `<text>`, checkpoint anterior vs este): **145.9→22.8 detecções/página, 94.7%→22.0% páginas com detecção** — queda grande, mas coerente com recall/precision TAMBÉM subindo no conjunto rotulado (não é um detector menos sensível de forma geral, parece estar discriminando melhor traço-de-arte vs caractere). Checkpoint anterior preservado em `weights/backups/detector_2026-07-18.pt`. |
| 2026-07-18 (rodada com bug, superada) | `weights/backups/detector_2026-07-18.pt` | ~17 imagens reais + tentativa de 300 páginas sintéticas — **mas o número de batches/epoch observado (~3.75, contra ~38 da rodada corrigida) sugere fortemente que o dado sintético NÃO foi incluído de verdade nesta rodada** (mesmo bug de resolução de diretório, não percebido na hora porque essa rodada não veio com o log de geração). Não dá pra confirmar com certeza retroativamente (log da geração não foi salvo). | mAP@50 0.72-0.74 | Superada pela rodada corrigida acima. Ganho sobre o baseline de 2026-07-10 provavelmente veio majoritariamente de hiperparâmetro/versão do YOLO (`yolo26n.pt`, imgsz 1024), não do dado sintético — registrado aqui só como hipótese, não fato confirmado. |
| 2026-07-10 | `weights/backups/detector_2026-07-10.pt` | ~17 imagens reais anotadas via Roboflow (`miguelmussalam/manga109-character-bouding-box`), sem nenhum dado sintético | mAP@50 ~0.64 (baseline) | Baseline original. Superado pelas duas rodadas acima. |

## Pipeline completo (validação corpus-wide, `src/helper/corpus_validate.py`)

Mesma base de comparação nas duas rodadas (`ground_truth_full.json`: 109 volumes, 8536 páginas com N1, **42611 pares esperados** — idêntico nas duas, confirma comparação limpa). Classificador não mudou entre as duas rodadas (`classifier_best.pt` de 2026-07-12) — a diferença de recall isola o efeito da troca de detector.

| Data | Detector usado | Recall agregado | OUTROS | Duração |
|---|---|---|---|---|
| 2026-07-19 | `best.pt` (rodada corrigida, mAP@50 0.81) | **82.69%** (35234/42611) | 88.1% | 93.5 min |
| 2026-07-15 | `detector_2026-07-18.pt`/anterior (mAP@50 ~0.64-0.74) | 77.99% (33233/42611) | 88.0% | ~92 min |

**+4.7pp de recall** atribuível à melhoria do detector (mAP 0.64→0.81), com o classificador congelado — evidência direta de que o ganho de mAP se traduz em recall real de pipeline, não só em métrica isolada do detector.

Por volume (109 volumes): média 82.9%, mediana 84.3%, desvio padrão 7.0. Pior: `LancelotFullThrottle` 62.2% (395/635). Melhores: `PikaruGenkiDesu`/`JijiBabaFight` 95.3%. Volumes citados na rodada anterior: `BokuHaSitatakaKun` (pior de todos antes, 51.1%) subiu pra **69.6%**; `PrayerHaNemurenai` (melhor antes, 94.3%) na verdade **caiu para 87.1%** — nem toda melhoria de detector é uniforme por volume, vale investigar esse caso se for relevante depois.

### Limitação conhecida desta medição
`corpus_validate.py` só agrega `hits`/`esperado` de `avaliar_pagina()` — não agrega `miss_detector`/`miss_classificador` (que a função já calcula, usado hoje só no benchmark pequeno de `benchmark.py`). Não dá pra saber, nesta rodada, quanto da melhoria veio de menos miss do detector especificamente vs. do classificador aproveitar melhor as caixas novas. Adicionar essa agregação seria um upgrade barato pra próxima rodada full-corpus.
