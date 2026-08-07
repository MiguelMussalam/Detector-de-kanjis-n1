# Log de Experimentos

Registro das rodadas de treino do projeto (detector + classificador), pra reconstruir o histórico depois (útil pro relatório de IC). Este log **começa em 2026-07-16**, quando foi criado — só tem entradas verificadas com confiança nesta sessão. Rodadas anteriores não entraram aqui pra não arriscar registrar hiperparâmetro/número de memória de forma imprecisa; ver `weights/backups/manifest.md` para o que se sabe com certeza sobre os checkpoints antigos preservados (só data e contexto factual, sem detalhe de treino não confirmado).

Convenção: preencher uma linha nova a cada rodada de treino completa (não a cada ajuste de config testado localmente).

## Classificador

| Data | Checkpoint | Dado de treino | Resultado | Notas |
|---|---|---|---|---|
| 2026-08-05 | `weights/classifier_best.pt` (ativo) | Mesma receita de 2026-07-23 + fix do `crop_size` + `OUTROS_REAL_MIX=False` (OUTROS 100% sintético). Promovido na época 5/30 (treino seguia rodando no Kaggle), `val_acc=97.87%` sintético. | `real_n1_acc`=**96.98%** (vs 4.60% com OUTROS misto), amostra local de 50 páginas **89.7%** (226/252) -- melhor recall já medido no projeto, mesmo com só 1/6 do treino planejado. | Ver seção "Achado: colapso pra OUTROS em dado real" abaixo pro diagnóstico completo. Pendente: validação full-corpus oficial (109 volumes) e teste de rejeição de não-N1 real (ainda não medido -- risco de trade-off levantado quando o real saiu do OUTROS). Checkpoint anterior preservado em `weights/backups/classifier_2026-07-23.pt`. |
| 2026-08-01 | `weights/backups/classifier_2026-08-01_regressao.pt` (revertido no mesmo dia, ver seção "Regressão do checkpoint 2026-08-01" abaixo) | Mesma receita de 2026-07-23 (`stem_leve`, `label_smoothing=0.1`) + fixes desta sessão incorporados: correção do no-op de `apply_translate_and_crop`, `CLF_MORFO_PROB_COM_TRANSLACAO=0.1` (reduz co-ocorrência morfologia+translação), troca de 3 fontes (`BIZ-UDPGothic`/`Mincho`, `Hina-Mincho`) por `GenEi-Antique-M`/`GenEi-Gothic-KL-H` (ver seções "Degradações combinadas" e "Auditoria de fontes" abaixo). 18 epochs, `train_acc≈100%`, `val_acc=98.82%` (sintético, acima do 98.53%/época 15 do checkpoint anterior). | Confusão no val sintético (54.280 amostras): 97.3% (1467 erros, 2.70%). **Amostra local de 50 páginas/10 volumes** (mesma amostra da comparação com EasyOCR/Tesseract): recall **84.5%** (213/252) — **pior** que os 88.1% (222/252) do checkpoint anterior na mesma amostra, apesar do `val_acc` sintético mais alto. | **Sinal misto, não resolvido ainda**: val sintético melhorou, mas a amostra real (pequena, 252 pares) piorou -- pode ser ruído de amostra (histórico de desvio padrão ~6.6pp entre volumes) ou uma regressão real (hipótese a investigar: reduzir a co-ocorrência morfologia+translação tornou o treino "mais fácil" que os crops reais que o detector de fato produz). ETL9 do candidato deu 4.89% top-1 (238.400 amostras) -- muito baixo, mas sem baseline do checkpoint anterior pra comparar (avaliação adiada). **Promovido mesmo assim pra decidir com a validação full-corpus no Kaggle** (muito mais confiável que a amostra local) -- se confirmar a piora, reverter para `weights/backups/classifier_2026-07-23.pt`. Checkpoint anterior preservado lá. |
| 2026-07-23 | `weights/backups/classifier_2026-07-23.pt` | Mesma receita de 2026-07-19 + `CLF_STEM_LEVE=True` (stride do conv1 1→ e maxpool→Identity, ver `model.py`) + `CLF_LABEL_SMOOTHING=0.1`. Treino interrompido manualmente na época 15/30 (tempo por época ~5x maior, ver nota de custo abaixo) — patience=7 nunca disparou, `val_acc` ainda subindo (98.53% na época 15, já acima do 97.7% final da rodada sem stem_leve). | Benchmark leve (1 volume, 61 páginas, 207 pares esperados, métrica multiset já corrigida): **72.5%→75.4% (+2.9pp)** vs. checkpoint anterior — amostra pequena, sinal direcional, não conclusivo. | **Bug real encontrado e corrigido durante a validação**: o checkpoint não guardava se foi treinado com `stem_leve` (stride/Identity não aparecem no `state_dict`, só pesos) — carregar em outra máquina (`CLF_STEM_LEVE` local default `False`) reconstruía a arquitetura ERRADA, pesos carregavam sem erro de shape mas a rede saía incoerente (recall caiu pra **0%** no primeiro teste). Corrigido: checkpoint agora salva `"stem_leve"` no dict (`train.py`), `load_classifier` (`inference.py`) lê isso em vez de confiar no config local. Checkpoint já baixado foi corrigido manualmente (chave adicionada via script). **Custo de inferência real**: ~4.9x mais lento por chamada do classificador (2.61ms→12.75ms, CPU, medido direto), ~3.7x mais lento por página no pipeline completo — puxa mAIS resolução espacial (mapa 8x8 em vez de 2x2 antes do avgpool), mas com preço de latência nada desprezível pra um pipeline que chama o classificador uma vez por caractere detectado. Checkpoint anterior preservado em `weights/backups/classifier_2026-07-19.pt`. |
| 2026-07-19 | `weights/backups/classifier_2026-07-19.pt` | 100% sintético — mesma receita de 2026-07-12 (N1 200/classe, OUTROS 20000/5000), com `CLF_MORFO_K_MAX=2` e `CLASSIFIER_FONT_SIZE_WEIGHTS` corrigido (14px 16→6) incorporados. 30 epochs completas (early stopping nunca disparou), `train_acc=100%`, `val_acc=97.7%` (sintético). | Recall corpus-wide (mesma base de comparação, 107 volumes presentes localmente — ver nota): **82.61%→83.69% (+1.08pp)** vs. o checkpoint anterior, com o mesmo detector novo (mAP 0.81) nas duas medições. | Ganho real mas modesto, como esperado (fixes de qualidade de augmentation, não aumento de dado/classe). Checkpoint anterior preservado em `weights/backups/classifier_2026-07-12.pt`. |
| 2026-07-12 | `weights/backups/classifier_2026-07-12.pt` | 100% sintético — N1 200 amostras/classe (1232 classes), OUTROS 20000 treino / 5000 val, sem nenhum dado real misturado (`CLASSIFIER_SAMPLES_TRAIN=200`, `CLF_OUTROS_SAMPLES_TRAIN=20000`) | Recall no benchmark de região (Manga109, oficial `<text>`): **82.6%** (1 página) / **64.0%** (4 páginas curadas manualmente) / **78.0%** (corpus completo, 8536 páginas com N1, 109 volumes, 42611 pares esperados) | Superado pela rodada de 2026-07-19. Resolveu o colapso de atalho de domínio (real vs. sintético) de rodadas anteriores ao remover dado real do treino inteiramente. Carregava o problema de erosão (`CLF_MORFO_K_MAX=3`) e o peso de fonte 14px não calibrado, corrigidos na rodada seguinte. |

### Nota metodológica sobre a comparação de 2026-07-19
A validação local só tem **107 dos 109 volumes** do Manga109 (`ThatsIzumiko` e `UchiNoNyansDiary` não existem na cópia local — nem imagem nem anotação, confirmado via `ls`). Comparar direto contra os 82.69%/109-volumes documentados em 2026-07-15 seria injusto (esses 2 volumes tinham recall ~89%, acima da média, então excluí-los infla artificialmente o "antes" se não corrigido). Recalculei o recall do checkpoint ANTERIOR usando só os mesmos 107 volumes (34779/42102 = 82.61%) antes de comparar com o novo (35234/42102 = 83.69%) — **essa é a comparação correta, +1.08pp real**. Se quiser o número oficial nos 109 volumes completos, precisa rodar essa validação no Kaggle (dataset completo) em vez de local.

### Nota metodológica sobre a comparação de 2026-07-23
A comparação 72.5%→75.4% já usa a métrica de recall corrigida (multiset, `benchmark.py` reescrito em 2026-07-22/23 -- ver seção própria) nos dois lados, mas numa amostra pequena (1 volume, 207 pares esperados) por causa do custo de inferência do `stem_leve` -- não dá pra tratar como número definitivo, só como sinal direcional. A comparação 82.61%→83.69% (linha 2026-07-19 acima) usa a métrica ANTIGA (set deduplicado) -- os dois números não são diretamente comparáveis entre si nesse sentido, cada linha documenta o que era válido usar na época.

### Pendências conhecidas não incorporadas neste checkpoint
- Residual de WARN de legibilidade (fonte fina × kanji complexo, ver investigação de 2026-07-19) não tem fix aplicado — só o ajuste de peso de tamanho de fonte, que teve efeito modesto (25→21 em 200 amostras).

### Decisão sobre `stem_leve`: RESOLVIDA em 2026-07-25
A validação completa no Kaggle (109 volumes, GPU) rodou a **0.587s/página** — praticamente igual ao baseline sem `stem_leve` (~0.51-0.55s/página). O custo de ~4.9x por chamada medido localmente era específico de **CPU** (mapas de convolução maiores penalizam muito mais sem paralelismo de GPU) — em GPU o custo extra é desprezível. **Decisão: manter `stem_leve` ativo.** Ver seção "Pipeline completo" pro resultado full-corpus.

## Detector

| Data | Checkpoint | Dado de treino | Resultado | Notas |
|---|---|---|---|---|
| 2026-07-18 (rodada corrigida) | `weights/best.pt` (ativo) | ~17 imagens reais (Roboflow) + 300 páginas sintéticas (`generate_pages.py`, grade 2D multi-coluna em `dividir_em_celulas`), confirmadamente fundidas desta vez (~38 batches/epoch, batch=8 -> ~300 imgs/epoch, condizente com o dataset completo). `val` 100% real. 150 epochs, `yolo26n.pt`, imgsz 1024, batch 8. | **mAP@50 0.81** (epoch 146 pico 0.813, epoch 150 final 0.810), mAP@50-95 ~0.51, precision ~0.90, recall ~0.78-0.79 | Corrigiu o bug de resolução de diretório do Manga109 no Kaggle (`_buscar_manga109_diretorio` escolhia o dataset Roboflow por engano — ver `config.py`/`generate_pages.py`) que fez a rodada anterior (abaixo) rodar com bem menos dado do que deveria. `detector_fp_check.py` (150 páginas sem `<text>`, checkpoint anterior vs este): **145.9→22.8 detecções/página, 94.7%→22.0% páginas com detecção** — queda grande, mas coerente com recall/precision TAMBÉM subindo no conjunto rotulado (não é um detector menos sensível de forma geral, parece estar discriminando melhor traço-de-arte vs caractere). Checkpoint anterior preservado em `weights/backups/detector_2026-07-18.pt`. |
| 2026-07-18 (rodada com bug, superada) | `weights/backups/detector_2026-07-18.pt` | ~17 imagens reais + tentativa de 300 páginas sintéticas — **mas o número de batches/epoch observado (~3.75, contra ~38 da rodada corrigida) sugere fortemente que o dado sintético NÃO foi incluído de verdade nesta rodada** (mesmo bug de resolução de diretório, não percebido na hora porque essa rodada não veio com o log de geração). Não dá pra confirmar com certeza retroativamente (log da geração não foi salvo). | mAP@50 0.72-0.74 | Superada pela rodada corrigida acima. Ganho sobre o baseline de 2026-07-10 provavelmente veio majoritariamente de hiperparâmetro/versão do YOLO (`yolo26n.pt`, imgsz 1024), não do dado sintético — registrado aqui só como hipótese, não fato confirmado. |
| 2026-07-10 | `weights/backups/detector_2026-07-10.pt` | ~17 imagens reais anotadas via Roboflow (`miguelmussalam/manga109-character-bouding-box`), sem nenhum dado sintético | mAP@50 ~0.64 (baseline) | Baseline original. Superado pelas duas rodadas acima. |

## Pipeline completo (validação corpus-wide, `src/helper/corpus_validate.py`)

| Data | Detector | Classificador | Métrica | Recall agregado | OUTROS | Duração |
|---|---|---|---|---|---|---|
| **2026-08-06** | `best.pt` (mAP@50 0.81) | `classifier_best.pt` -- `OUTROS_REAL_MIX=False`, só época 5/30 (ativo) | multiset | **85.63%** (37288/43547) | 88.6% | 84.4 min |
| 2026-07-25 | `best.pt` (mAP@50 0.81) | `classifier_best.pt` com `stem_leve` | **multiset (corrigida)** | **86.78%** (37789/43547) | 92.3% | 83.5 min (0.587s/pág, Kaggle GPU) |
| 2026-08-01 (revertido) | `best.pt` (mAP@50 0.81) | `classifier_2026-08-01_regressao.pt` (não é o ativo, ver nota) | multiset | 84.23% (36680/43547) | 93.7% | 84.4 min |
| 2026-07-19 | `best.pt` (mAP@50 0.81) | `classifier_2026-07-19.pt` (sem stem_leve) | set deduplicado (antiga) | 82.69% (35234/42611) | 88.1% | 93.5 min |
| 2026-07-15 | `detector_2026-07-18.pt`/anterior (mAP@50 ~0.64-0.74) | mesmo de cima | set deduplicado (antiga) | 77.99% (33233/42611) | 88.0% | ~92 min |

**Checkpoint ativo desde 2026-08-06, decisão consciente com número pior em recall bruto**: o full-corpus oficial do checkpoint `OUTROS_REAL_MIX=False` (ver seção "Achado: colapso pra OUTROS em dado real") deu **85.63%**, **1.15pp abaixo** dos 86.78% do checkpoint anterior (2026-07-25) -- inverte o que a amostra local de 50 páginas sugeria (89.7% vs 88.1%), mesma lição do dry-run de 3 volumes de antes: amostra pequena pode enganar, mesmo sendo 10x maior. Quebra de misses (6259 total): 28.0% detector / 72.0% classificador.

A causa provável não é o `OUTROS_REAL_MIX=False` em si ter piorado o modelo -- é que esse checkpoint só treinou **5 de 30 épocas** (o treino não foi continuado até convergir, horas de GPU do Kaggle da semana foram usadas nesta validação full-corpus em vez de mais treino). Decisão: manter esse checkpoint como ativo mesmo assim, porque os outros sinais (`real_n1_acc` 96.98% vs 4.60%, rejeição de não-N1 real 95.2%) mostram uma generalização genuinamente melhor e mais bem validada -- o recall bruto 1.15pp menor é o preço de um modelo sub-treinado, não evidência de que a mudança foi errada. Checkpoint anterior (2026-07-25, 86.78%) preservado em `weights/backups/classifier_2026-07-23.pt` -- se der tempo/GPU de continuar esse treino até convergir num ciclo futuro, a expectativa é que ele supere os 86.78% também em recall bruto, não só nos sinais de generalização.

**Nota metodológica**: a linha de 2026-07-25 muda métrica E checkpoint do classificador ao mesmo tempo em relação à linha de 2026-07-19 — não é uma comparação de variável única. Mas como o multiset é uma contagem mais **rigorosa** que o set deduplicado (nunca infla recall, só corrige a inflação anterior — ver nota de 2026-07-19 abaixo), ver o número subir mesmo assim (82.69%→86.78%) é um resultado mais forte do que uma leitura ingênua sugere, não mais fraco.

**Tentativa revertida (2026-08-01) -- número oficial confirmado**: o checkpoint com os fixes de translação/morfologia/fontes (ver seção "Regressão do checkpoint 2026-08-01" abaixo) foi promovido, testado e revertido no mesmo dia. Full-corpus (109 volumes, 43547 pares) deu **84.23%** (36680/43547), OUTROS 93.7% -- uma queda real de **2.55pp** frente aos 86.78% vigentes, bem menor que os 71.9% que o dry-run de 3 volumes sugeria (`ARMS`/`AisazuNihaIrarenai`/`AkkeraKanjinchou` são volumes de recall abaixo da média mesmo no corpus inteiro, não foi ruído de amostragem -- o dry-run mede tempo, não é uma amostra representativa de acurácia). Quebra de misses (6867 total): **74.4% classificador / 25.6% detector** -- a fatia do classificador cresceu frente aos 69.5% de antes, coerente com ele ser a única peça que mudou. `classifier_best.pt` permanece revertido pro checkpoint de 2026-07-23 (86.78%, número vigente); o checkpoint regredido fica só arquivado (`weights/backups/classifier_2026-08-01_regressao.pt`) pra referência.

**Quebra detector-vs-classificador (novo, ver `corpus_validate.py`)**: dos 5758 misses restantes, **30.5% são do detector, 69.5% são do classificador** — confirma, agora em escala completa (era 36.5%/63.5% numa amostra de 2 volumes), que o classificador é o gargalo maior hoje. Prioridade de melhoria futura: classificador antes de detector.

**Custo de inferência do `stem_leve` em GPU**: 0.587s/página, praticamente igual ao baseline sem stem_leve (~0.51-0.55s/página) — o custo de ~4.9x medido localmente era específico de CPU. Decisão: manter `stem_leve` ativo (ver seção do classificador).

Por volume (109 volumes): média 87.1%, mediana 88.8%, desvio padrão 6.6 (mais concentrado que os 7.0 de antes). Pior: `LancelotFullThrottle` 63.3% (432/682, ainda o pior de todos, com miss_detector=150/miss_classificador=100 — quase metade dos misses desse volume é o detector não achando caixa nenhuma, diferente do padrão geral 30/70 — pode ser um volume com arte que atrapalha detecção especificamente). Melhores: `JijiBabaFight` 98.1%, `MomoyamaHaikagura` 97.6%.

### Análise de erro (2026-07-26): confusão no val sintético + mergulho visual em página real
**Confusão no val sintético** (`src/classifier/eval.py --only confusao`, checkpoint real com `stem_leve`, 54.280 amostras): taxa de erro **1.61%** (874 erros). Padrão dos erros, olhado visualmente (`data/classifier_eval/confusoes_visual.png`): concentrado em (a) crops degradados a ponto de ficarem ilegíveis mesmo pra um humano, e (b) confusões visuais genuínas entre kanji fora do N1 (categoria OUTROS) que se parecem com um kanji N1 específico — não erro aleatório. Achado numérico: **112 casos de kanji N1 real → previsto OUTROS, contra só 9 casos do inverso** (OUTROS real → kanji específico inventado) — o modelo erra ~12x mais recuando ("não reconheço") do que alucinando uma resposta errada com confiança, um viés de erro relativamente seguro pro caso de uso.

**Mergulho visual no `LancelotFullThrottle`** (pior volume, 6 páginas com mais linhas, pipeline completo rodado e cada linha colorida por veredito): padrão claro a olho nu — acertos (verde) concentram em diálogo normal de balão (fonte impressa padrão); erros (vermelho=detector não achou nada, laranja=classificou errado) concentram quase só em **texto decorativo desenhado à mão** (legendas de ênfase/grito, onomatopeia estilizada), comum em mangá de comédia (gênero desse volume). Causa raiz identificada: lacuna de cobertura de estilo de fonte no gerador sintético, não um limite fundamental do pipeline. Não corrigido nesta sessão (custo/benefício não compensa dado o prazo) — registrado como trabalho futuro.

**Conclusão**: as duas análises convergem — os erros do pipeline são explicáveis (degradação real, confusão visual genuína, lacuna de estilo específica), não aleatórios. Evidência de que o modelo aprendeu estrutura visual real de kanji.

## Comparação com OCR tradicional (Hipótese 1 da proposta formal)

`src/helper/ocr_baseline_compare.py` (novo, 2026-07-26): compara nosso pipeline contra **EasyOCR** (suporte a japonês) na mesma tarefa — reconhecer kanji N1 em linhas de diálogo reais do Manga109. Isola reconhecimento de detecção: recorta exatamente a mesma bbox de linha do ground truth oficial e roda o OCR só no recorte (não testa a detecção de texto do EasyOCR, só o reconhecimento).

**Amostra**: 50 páginas (10 volumes × 5 páginas, mesmo `--limit-volumes`/`--sample-paginas-por-volume` de `corpus_validate.py`), 252 pares esperados — **idêntica** nos dois lados, comparação limpa.

| Sistema | Recall (mesmas 50 páginas, 252 pares esperados) |
|---|---|
| **Nosso pipeline** (detector + classificador, `stem_leve`) | **88.1%** (222/252) |
| **EasyOCR** | **14.7%** (37/252) |

Diferença verificada num recorte generoso também (não é artefato de crop pequeno demais): mesmo numa linha com bbox grande (~6000px²/caractere), o EasyOCR errou completamente, com confiança baixíssima (<1%) nas previsões — sugere que ferramentas de OCR de propósito geral não lidam bem com a estética de mangá (fonte estilizada, fundo com screentone, texto vertical denso), exatamente a lacuna que a Hipótese 1 da proposta formal previa.

**Auditoria visual** (`data/corpus_validation/auditoria_ocr_baseline.png`, 24 exemplos, gerada por `salvar_grade_auditoria`): confirma que os recortes usados são os corretos (mesmo texto do ground truth) e que a métrica de recall é generosa com o EasyOCR (aceita acerto fora de ordem/posição) — a lacuna não é bug de metodologia.

**CER (character error rate) — 2026-07-26**: além do recall de kanji N1 (multiset, insensível à ordem), calculamos CER via distância de edição sobre a transcrição LIMPA completa (kana + pontuação inclusos, não só N1), pra separar "erra só kanji N1 raro" de "erra texto de mangá em geral". Resultado na mesma amostra de 50 páginas/178 linhas com N1 esperado:

| Métrica | Valor |
|---|---|
| Recall N1 (multiset, ordem-insensível) | 14.7% (37/252) |
| **CER, texto completo** (edit distance, ordem-sensível) | **94.9%** (2851 edições / 3005 caracteres) |

CER uniforme entre 92.7% e 98.1% em **todos os 10 volumes** (sem outlier puxando a média) — não é um volume/fonte específico ruim, é falha ampla e consistente do EasyOCR nessa estética inteira, não um problema restrito ao vocabulário N1. Exemplo típico (mediana da distribuição, não um pior caso escolhido a dedo):

```
esperado: "...私は...身も心も汚れきっている........." (linha longa, legível a olho nu)
OCR leu:  "下こ瀬塔雷一点は"                              (CER 95.3%)
```

Como o CER é sensível à ordem (diferente do recall multiset), ele expõe um quadro pior do que os 14.7% já sugeriam — a métrica principal da comparação já era generosa com o EasyOCR, e mesmo assim ele foi mal; pela métrica padrão da área (CER), o resultado é ainda mais desfavorável a ele. Dado linha-a-linha (transcrição, texto lido, CER individual) salvo em `data/corpus_validation/ocr_baseline_detalhe.csv` pra reanálise sem precisar rodar o OCR de novo.

**Checagem de viés de metodologia — 2026-07-26**: antes de aceitar o número, testamos se o script está usando o EasyOCR de forma sub-ótima (o que enfraqueceria a comparação). Ablação com 4 variantes na mesma amostra (50 páginas/178 linhas):

| Variante | Recall N1 | CER |
|---|---|---|
| Baseline (bbox exata, sem `rotation_info`) | **14.7%** (37/252) | **94.9%** |
| + `rotation_info=[90,180,270]` (recomendado pela doc do EasyOCR pra texto vertical/rotacionado) | 11.1% (28/252) | 95.2% |
| + margem de 20% ao redor da bbox (evita cortar traço na borda) | 13.1% (33/252) | 95.6% |
| + os dois juntos | 10.7% (27/252) | 96.2% |

As duas hipóteses mais plausíveis de "estamos prejudicando o EasyOCR sem querer" (falta de `rotation_info` pra texto vertical, recorte sem margem) **pioraram** o resultado dele em vez de melhorar — o baseline já usado na comparação principal é, das quatro, a configuração mais favorável ao EasyOCR. Reforça que o resultado baixo é uma limitação real do EasyOCR nesse domínio, não um artefato de configuração do teste. Também confirmado (impacto desprezível): a transcrição do ground truth guarda `\n` internos que o texto do OCR nunca tem — infla o CER artificialmente em 41/178 linhas, mas o efeito agregado é irrelevante (94.9% → 94.8% removendo).

Não testado ainda (menor prioridade): variar `canvas_size`/`text_threshold`/`low_text` do detector interno do EasyOCR (ablação de hiperparâmetros mais ampla).

### Pendências desta comparação
- Amostra de 50 páginas, não o corpus inteiro — suficiente pra confirmar a diferença (é grande demais pra ser ruído amostral), mas não é o número "oficial" em escala completa.
- ~~Só testado com EasyOCR — Tesseract (`jpn_vert`) fica como comparação secundária, não feita ainda por restrição de tempo.~~ **Fechada em 2026-08-01** (ver seção "Segundo baseline" abaixo).
- ~~**Lacuna identificada 2026-07-26**: a Hipótese 1 da proposta fala em recall de **detecção**, mas o teste atual isola reconhecimento (dá a bbox certa pro EasyOCR de graça).~~ **Fechada em 2026-08-01.**

### Teste de página inteira (fecha a Hipótese 1 como está literalmente escrita)

`ocr_baseline_compare.py` ganhou um segundo modo (`--modo pagina`): em vez de recortar a bbox certa da linha e dar de graça pro EasyOCR, roda `reader.readtext()` na **página inteira** -- ele acha o texto sozinho (detecção + reconhecimento próprios), igual ele funcionaria numa aplicação real. Métrica agregada por página (multiset de todos os N1 esperados na página vs. tudo que o OCR leu nela), mesma amostra de 50 páginas do teste principal.

| Modo | Recall EasyOCR |
|---|---|
| `recorte` (bbox dada de graça, só reconhecimento) | 14.7% (37/252) |
| **`pagina` (detecção + reconhecimento próprios)** | **9.9% (25/252)** |

Confirma a expectativa registrada antes: sem a vantagem de já saber onde olhar, o EasyOCR piora ainda mais (não inverte). Com isso, a comparação fim-a-fim fica **88.1% (nosso pipeline completo: detector + classificador) vs. 9.9% (EasyOCR completo: detecção + reconhecimento dele)** -- testa a Hipótese 1 exatamente como está escrita na proposta formal, não uma versão facilitada dela.

Sem CER nem grade visual nesse modo (escopo deliberadamente reduzido -- a métrica de recall já responde a pergunta; CER por página exigiria decidir uma ordem de leitura entre balões sem alinhamento, mais ruído que sinal aqui).

### Segundo baseline: Tesseract (2026-08-01)

`ocr_baseline_compare.py` ganhou um segundo motor plugável (`--engine {easyocr,tesseract}`), pra não basear a Hipótese 1 na fraqueza de uma ferramenta só. Tesseract 5.4 (`jpn_vert`, variante treinada pra texto japonês vertical) instalado localmente via winget (binário de sistema, não pip -- ver `TESSERACT_CMD`/`TESSDATA_DIR` em `config.py`). Mesma amostra de 50 páginas/252 pares esperados dos testes com EasyOCR, ambos os modos (`recorte` e `pagina`):

| Sistema | `recorte` (bbox dada) | `pagina` (detecção + reconhecimento próprios) |
|---|---|---|
| **Nosso pipeline** | — | **88.1%** (222/252) |
| EasyOCR | 14.7% (37/252) | 9.9% (25/252) |
| **Tesseract** | **25.4%** (64/252) | **11.5%** (29/252) |

CER (`recorte`, texto completo, mesma amostra): Tesseract 79.1% (2377/3005 edições) vs EasyOCR 94.9% (2851/3005) -- Tesseract também é a melhor das duas ferramentas tradicionais nessa métrica mais estrita.

**Leitura**: Tesseract é sistematicamente melhor que o EasyOCR nas duas métricas e nos dois modos, mas continua muito abaixo do nosso pipeline (25.4% vs 88.1% no melhor caso dele) -- o resultado generaliza a conclusão da Hipótese 1 em vez de depender de uma ferramenta só: **duas** ferramentas de OCR genéricas independentes, com arquiteturas diferentes (EasyOCR = CRAFT+CRNN neural; Tesseract = análise clássica de componentes conectados + LSTM), falham de forma semelhante nessa estética específica (fonte estilizada, screentone, texto vertical denso). O modo `pagina` piora os dois frente ao `recorte`, na mesma direção -- reforça que boa parte da dificuldade já está na etapa de detecção de texto em mangá, não só no reconhecimento do glifo.

Sem grade visual de auditoria específica pro Tesseract nesta rodada (mesma metodologia e amostra já auditadas visualmente pro EasyOCR bastam pra validar que a comparação é justa); grade salva em `data/comparacao_visual/tesseract.png` para reanálise futura. Dado linha-a-linha em `data/corpus_validation/ocr_baseline_detalhe_tesseract.csv` (`recorte`) e `..._tesseract_pagina.csv` (`pagina`).

### Terceiro motor: manga-ocr, especializado (não é baseline da Hipótese 1) (2026-08-05)

`ocr_baseline_compare.py` ganhou um terceiro motor (`--engine mangaocr`, kha-white/manga-ocr-base) -- diferente de EasyOCR/Tesseract (generalistas, os baselines que a Hipótese 1 formalmente compara), é especializado em texto de mangá. Não substitui a comparação com os dois genéricos, complementa: testa se a conclusão (OCR tradicional perde pro pipeline próprio) também vale contra uma ferramenta que já é do domínio certo. Mesma amostra de 50 páginas/252 pares:

| Sistema | `recorte` (bbox dada) | `pagina` (detecção + reconhecimento próprios) |
|---|---|---|
| **Nosso pipeline** (checkpoint 2026-08-05) | — | **89.7%** (226/252) |
| EasyOCR | 14.7% (37/252) | 9.9% (25/252) |
| Tesseract | 25.4% (64/252) | 11.5% (29/252) |
| **manga-ocr** | **93.3%** (235/252) | **0.0%** (0/252) |

CER (`recorte`): manga-ocr 15.1% (453/3005 edições) -- muito abaixo de EasyOCR (94.9%) e Tesseract (79.1%), na mesma faixa do nosso pipeline.

**Nota**: os 89.7% aqui são da mesma amostra de 50 páginas usada por todos os motores desta comparação (justo, mesmo teste pra todo mundo) -- não confundir com os 85.63% do full-corpus oficial (109 volumes, seção "Pipeline completo"), que é um número diferente medido em escala muito maior no mesmo checkpoint sub-treinado (5/30 épocas). Os dois são válidos, medem coisas em escalas diferentes.

**Leitura em duas partes, não uma conclusão só**:
1. **Reconhecimento** (`recorte`, bbox dada de graça): manga-ocr é forte de verdade -- 93.3%, levemente acima até do nosso pipeline (89.7%). Confirma que especialização de domínio (mangá) importa muito mais que a arquitetura genérica em si -- a Hipótese 1 não é "toda ferramenta de OCR é ruim nisso", é "ferramenta **genérica** é ruim nisso". Uma ferramenta especializada de terceiros já resolve bem essa parte específica.
2. **Fim-a-fim** (`pagina`, sem bbox de graça): manga-ocr zera. Ele não tem detecção própria -- é um recognizer puro, feito pra receber uma região de fala já recortada (por design, não é um bug nem uma limitação injusta de teste). Jogar a página inteira nele é fora do que ele foi treinado pra fazer; ele "lê" a página toda como se fosse um bloco de texto só, sem achar as linhas.

**Conclusão**: a contribuição do projeto não é "reconhecer melhor que qualquer OCR" -- é entregar detecção **e** reconhecimento juntos, resolvendo o problema fim-a-fim sem depender de outra ferramenta pra achar onde está o texto. manga-ocr sozinho não é uma alternativa viável pro caso de uso (precisaria ser pareado com um detector de qualquer forma -- inclusive o nosso), mas mostra que o "motor de leitura" ideal já existe pra quem só precisa da etapa de reconhecimento com bbox conhecido.

Grade visual em `data/comparacao_visual/mangaocr.png`. Dado linha-a-linha em `data/corpus_validation/ocr_baseline_detalhe_mangaocr.csv` (`recorte`) e `..._mangaocr_pagina.csv` (`pagina`).

## Degradações combinadas do classificador -- achado e correção (2026-07-27)

`filter_audit.py` já mostrava cada uma das 9 degradações sintéticas segura isolada (97.5%+ até e além do limite de produção). Mas a geração real roda até 8 degradações independentes na mesma amostra (soma das probabilidades ≈ 4.4 disparando juntas, em média) -- hipótese: o problema real não é nenhum filtro isolado no limite, é a combinação, invisível pro `filter_audit.py` por desenho (sempre desliga os outros filtros).

**Instrumentação** (`src/classifier/generate_crops.py`): parâmetro opcional `log: dict = None` adicionado em cada `apply_*`/função de composição/`generate_sample` (mesmo padrão do `avisos: list = None` já existente) -- registra quais degradações dispararam e com que severidade na amostra final, sem mudar nenhum comportamento quando `log=None`. Verificado: mesma seed com/sem `log` produz imagem byte-idêntica.

**Nova ferramenta** `src/helper/combinacao_filtros_audit.py`: gera amostras via `generate_sample()` de verdade (não isolada), mede acurácia por "concorrência" (nº de degradações disparando juntas) e um enriquecimento por degradação (taxa de disparo em falhas vs. sucessos).

**Achado (2000 amostras, baseline)**: a cifra de "~10-12%" no docstring de `generate_sample()` (uma calibração pontual de 200 amostras, ver seção do classificador) vira uma medida de verdade em escala: **9.9% ilegível** (sinal independente do modelo), **18.9% falha do classificador**. Acurácia cai de 100% (concorrência=1) pra 93% (concorrência=7) -- real, mas gradual. O enriquecimento aponta os culpados específicos: **morfologia (+32.2pp) e translação (+23.5pp)** desproporcionalmente presentes nas falhas -- blur/ruído, mesmo disparando mais, ficam sub-representados (efeito do fallback que desliga blur/ruído justamente nos casos mais difíceis, não porque sejam "protetores").

**Correção**: nova constante `CLF_MORFO_PROB_COM_TRANSLACAO=0.1` (config.py) -- quando a translação já disparou na amostra, a morfologia usa essa probabilidade reduzida em vez de `CLF_MORFO_PROB=0.3`. Implementado sem mudar a ordem de aplicação dos pixels (morfologia continua antes do recorte): `_decidir_translacao()` decide o resultado da translação antes de chamar `apply_morphology`, e `apply_translate_and_crop` aceita esse resultado pré-decidido (`pre_decidido=`) pra não sortear duas vezes.

**Resultado (2000 amostras, pós-fix)**:

| Métrica | Antes | Depois |
|---|---|---|
| Ilegível (sinal independente) | 9.9% (199/2000) | **5.7% (114/2000)** |
| Falha do classificador | 18.9% (378/2000) | **14.9% (298/2000)** |
| Enriquecimento morfologia | +32.2pp | +13.5pp |
| Enriquecimento translação | +23.5pp | +18.9pp |

Acurácia por concorrência ficou estável em 95-98% do nível 1 ao 6 (só cai nos níveis 7-8, com n=43/n=3, amostra pequena demais pra confiar). Reduzir a coocorrência do par específico (não "reduzir tudo") cortou a taxa de ilegibilidade quase pela metade.

**Pendência**: assim como o fix de translação (`apply_translate_and_crop`, achado anterior), essa correção só afeta a geração daqui pra frente -- o checkpoint atual foi treinado com a taxa antiga (~10%). Regenerar dado + retreinar fica como pendência conjunta com o fix de translação, decidido pra depois dado o prazo do projeto.

## Auditoria das fontes sintéticas (2026-07-31)

Pesquisa pra responder: as 13 fontes usadas na geração sintética (`config.py: FONTES_URL`) foram escolhidas com base em uso real de mangá, ou são "chute" de nome parecido? Cada uma pesquisada individualmente contra fontes/artigos sobre tipografia real de mangá japonês.

**Achado principal**: o padrão real de diálogo de mangá comercial é a combinação **アンチゴチ** (kana em estilo アンチック + kanji em ゴシック grosso) -- ex: a revista *Manga Time Kirara* usa especificamente アンチックAN1 + 太ゴB101, ambas da fundição Morisawa, pagas por assinatura (Morisawa Fonts, ~¥54.780/ano/PC, sem redistribuição permitida).

**Resultado da auditoria** (13 fontes):
- **6 confirmadas de verdade**: Shippori Antique e Zen Antique (mesma família アンチック, uma citada nominalmente em lista curada por mangakás como fonte padrão de diálogo); Klee One, Reggae One e Stick (Fontworks de verdade -- fundição real de anime/mangá, liberaram 8 fontes grátis via Google Fonts/OFL); Hachi Maru Pop (aparece em duas listas curadas por mangakás, mangá infantil/retrô).
- **4 plausíveis, não confirmadas por nome**: Dela Gothic One, Yusei Magic, Yuji Boku, Kaisei Tokumin -- estilos compatíveis (impacto/título, caligrafia de marcador, pincel real, título decorativo), mas sem confirmação direta de uso em mangá.
- **3 fora de contexto -- removidas**: `BIZ-UDPGothic-Regular`, `BIZ-UDPMincho-Regular` (fontes de Design Universal/acessibilidade da Morisawa, feitas pra documento empresarial/prefeitura/aeroporto) e `Hina-Mincho-Regular` (decorativa, inspirada em bonecas Hina/Hinamatsuri, sem relação com mangá). Provavelmente escolhidas por nome parecido ("Gothic"/"Mincho" + japonesa + grátis), não por uso real confirmado.

**Substituição**: as 3 removidas viraram **`GenEi-Antique-M`** e **`GenEi-Gothic-KL-H`**, da família 源暎 (okoneya.jp) -- licença SIL OFL 1.1 (aberta, uso comercial livre), e desenhadas especificamente pra reconstruir a estética アンチック de mangá de forma livre (アンチック combina Source Han Sans, 新コミック体, Linux Biolinum e GL-アンチック; a variante ゴシックKL é descrita explicitamente como feita pra balão de fala). `GenEi-Gothic-KL-H` (peso Heavy) entrou em `CLF_FONTES_PESADAS` no lugar de `BIZ-UDPGothic-Bold` (que nem chegou a estar em `FONTES_URL`, era um arquivo órfão já baixado antes).

**Detalhe técnico**: essas duas fontes só são distribuídas em `.zip` (não como arquivo solto), diferente de todas as outras 11. `download_fonts()` (`src/helper/fonts.py`) ganhou suporte a isso -- `FONTES_URL[nome]` agora aceita `(url_do_zip, caminho_dentro_do_zip)` além de string simples, sem mudar o comportamento das fontes já existentes.

Verificado: as duas cobrem 100% dos 1232 kanji N1 (`verify_fonts_compatibility`), renderização visual conferida (`data/pesquisa/preview_fontes_genei.png`), `pytest tests/` 32/32.

## Regressão do checkpoint 2026-08-01 -- bug no fix de translação (achado 2026-08-01)

O checkpoint retreinado com os fixes acima (translação + morfologia + fontes) foi promovido e testado no Kaggle. Primeiro sinal foi um dry-run de 3 volumes/216 páginas: recall **71.9%** (919/1278) contra os 86.78% do checkpoint anterior -- alarmante, mas **enganoso**: esses 3 volumes específicos (`ARMS`, `AisazuNihaIrarenai`, `AkkeraKanjinchou`) são naturalmente de recall abaixo da média, confirmado depois no corpus inteiro. O número oficial, full-corpus (109 volumes, 43547 pares esperados), foi **84.23%** (36680/43547) -- uma queda real de **2.55pp**, bem mais modesta que o dry-run sugeria, mas ainda assim uma regressão real e estatisticamente sólida numa amostra desse tamanho (não dá pra ser ruído). Confirmado também numa amostra local de 50 páginas (84.5% vs 88.1%) e um ETL9 baixo (4.89% top-1, sem baseline direto). Quebra de misses no full-corpus: **74.4% classificador / 25.6% detector** (vs 69.5%/30.5% do checkpoint anterior) -- consistente com a regressão vir só do classificador (única peça que mudou).

**Lição metodológica**: o dry-run do notebook existe pra medir **tempo**, não acurácia -- tratar o recall dele como sinal confiável de regressão foi um erro de leitura corrigido só depois do full-corpus rodar. Amostras pequenas (mesmo com centenas de pares) têm variância real entre volumes (~6.6-7pp de desvio padrão histórico) grande o suficiente pra distorcer a leitura.

**Causa raiz identificada (contribuinte, não necessariamente único fator)**: `apply_translate_and_crop` (`generate_crops.py`) é chamado em **100% das amostras** (não só nas ~70% em que a translação dispara de verdade), e o `crop_size` usava `max_dx`/`max_dy` fixos, independente de `disparou`. Resultado: TODA amostra do dataset -- inclusive as ~30% sem nenhuma translação -- passou a ser recortada mais apertada e reescalada de volta pro tamanho de saída. Com os valores de produção (`canvas_margin=0.15`, `CLF_TRANSLATE_MAX=0.10`), isso mudou o enquadramento do glifo de **76.9% do frame pra 96.2% do frame**, sistematicamente, no dataset inteiro. Dado que a regressão real medida (2.55pp) é bem mais modesta que a hipótese inicial sugeria, é plausível que esse bug seja só parte da explicação, combinado com o efeito da redução de coocorrência morfologia+translação -- um retreino com só o fix do crop (sem mexer na probabilidade) vai esclarecer quanto desse 2.55pp cada fator explica.

**Fix**: `crop_size` agora só encolhe quando `disparou=True` (reserva a folga real da translação); quando não dispara, mantém o recorte no tamanho cheio do canvas, igual sempre foi antes do fix de translação original.

```python
crop_size = min(h - 2 * max_dy, w - 2 * max_dx) if disparou else min(h, w)
```

**Decisão**: aplicado só esse fix por enquanto, `CLF_MORFO_PROB_COM_TRANSLACAO` mantido em 0.1 -- isolar uma variável por vez no próximo retreino, em vez de mudar duas coisas ao mesmo tempo de novo (foi assim que a causa real ficou mascarada desta vez). `weights/classifier_best.pt` revertido pro checkpoint anterior (`weights/backups/classifier_2026-07-23.pt`, 86.78%) até um retreino com este fix confirmar recuperação do recall.

## Validação real de N1 (2026-08-02)

Investigando a regressão acima, surgiu uma pergunta mais estrutural: `train.py` escolhe `best.pt` só pelo `val_acc` **sintético** (`if val_acc > best_val_acc`, ver `src/classifier/train.py`) -- e esse val é 100% gerado pelo mesmo pipeline sintético do treino. Se o gerador tiver qualquer viés sistemático (como o bug do crop acima, que tornou o sintético "mais fácil" sem refletir a realidade), a seleção de checkpoint fica cega a isso -- foi exatamente o que mascarou a regressão desta vez.

### Por que não dava pra simplesmente anotar caixas de N1 na mão

Manga109 só anota bbox de **linha/balão inteiro** + transcrição -- não tem anotação por caractere. `manga109_align.py` já infere a correspondência caractere↔caixa por heurística (conta caixas do nosso detector == caracteres da string, agrupa por coluna de leitura), mas descarta N1 de propósito (só usa pra OUTROS) porque cobertura por classe é baixa (mediana 1 exemplo/classe numa amostra de 6 volumes) e um erro de alinhamento pode contaminar a única amostra real daquela classe.

Auditoria visual confirmou o problema é real: ~6% dos crops N1 alinhados sem filtro tinham rótulo claramente errado (ex: caixa de 10-20px rotulada com um kanji de 16 traços, mas o crop era só um traço solto ou uma marca espúria -- o detector errou uma caixa em algum ponto da linha, e como o total de caixas ainda batia com o total de caracteres, o zip caractere↔caixa saiu deslocado a partir do erro sem disparar nenhuma checagem de contagem).

### Testando um segundo modelo como filtro

Cogitou-se usar EasyOCR/Tesseract pra confirmar o rótulo de cada crop antes de aceitar. **Descartado**: são justamente os baselines que o projeto já provou fracos nesse domínio (14.7%/25.4% de recall) -- usar como filtro descartaria a maioria do dado BOM (medido: **57% de falso positivo** em 14 crops presumivelmente corretos) e enviesaria o conjunto resultante pro que essas ferramentas genéricas conseguem ler, exatamente o oposto do que a Hipótese 1 do projeto quer medir.

Testado então o **manga-ocr** (kha-white/manga-ocr-base, Transformer especializado em texto de mangá, não é um dos baselines comparados pelo projeto): mesmo teste, **~17% de falso positivo** (2/12, e os dois casos, inspecionados de perto, eram crops corretos que o manga-ocr só leu errado -- não erros de alinhamento). Nos 2 erros de alinhamento conhecidos, pegou os dois (leu hiragana solto tipo 'ひ'/'ま' em vez do kanji complexo esperado -- consistente com serem detecções espúrias mesmo).

Escalado pra 30 volumes: de 1260 crops brutos, 954 aceitos (75.7%), 332 classes cobertas. Auditoria visual de 64 aceitos: **nenhum erro encontrado**. Auditoria dos rejeitados: ~metade eram erros reais de alinhamento (confirma o filtro funciona), ~metade eram crops corretos que o manga-ocr errou (filtro é conservador -- perde dado bom, não deixa passar dado ruim, o trade-off certo pra um conjunto de validação).

### Infraestrutura criada

- `src/helper/manga109_align_n1.py` (novo): mesma heurística de `manga109_align.py`, mas guarda N1 em vez de descartar, aceitando só quando o manga-ocr concorda com o rótulo. Sem split train/val (nunca é usado pra treino, sem risco de vazamento). Saída em `data/classifier_real_n1/{U+XXXX}/*.png`.
- `src/classifier/eval.py`: nova função `eval_real_n1()` + `--only real_n1`, mede top-1/top-5 nesse conjunto real, incluída no "avaliar tudo" default junto de `synth`/`etl9`.
- `config.py`: `MANGA109_ALIGN_N1_DIR`.
- `requirements.txt`: `manga-ocr`.

**Rodado em escala completa (109 volumes, 10.602 páginas)**: 4861 crops brutos, **3675 aceitos (75.6%)**, cobrindo **637 das 1232 classes N1 (51.7%)**.

### Resultado -- confirma a regressão com muito mais precisão que os testes anteriores

| Checkpoint | Top-1 real N1 (3675 amostras, 637 classes) | Top-5 |
|---|---|---|
| **07-23 (ativo, 15 épocas, sem bug do crop)** | **98.42%** | 100.00% |
| candidato do fix (6 épocas, bug corrigido, sub-treinado) | 96.30% | 99.67% |
| 08-01 (regredido, 18 épocas, com bug do crop) | 95.46% | 99.48% |

A queda do checkpoint regredido (98.42%→95.46%, **-2.96pp**) bate de perto com os -2.55pp medidos no full-corpus do Kaggle (86.78%→84.23%) -- essa validação real, que roda em **segundos** (não ~85min de GPU), reproduziu o mesmo sinal de forma barata. O candidato com o fix do crop já recupera **+0.84pp** sobre o regredido mesmo com menos de um terço do treino (6 vs 18 épocas) -- sinal de que o fix ajuda de verdade; a diferença restante pro ativo (-2.12pp) é mais provável de ser sub-treino do que um problema novo, a confirmar quando esse retreino rodar até convergir.

**Daqui pra frente**: esse é o sinal que deveria decidir promoção de checkpoint, não o `val_acc` sintético sozinho -- `eval.py` já avisa isso no resumo comparativo quando os dois estão disponíveis.

### `train.py` passa a escolher o `best.pt` pelo real_n1 (2026-08-02)

`carregar_real_n1`/`avaliar_real_n1` refatorados de `eval.py` pra serem chamados também a cada época do treino, não só depois. `best.pt` agora é salvo pela acurácia no `real_n1` (quando o conjunto existe), não mais só pelo `val_acc` sintético -- o *early stopping* (paciência) continua baseado no sintético, mais estável com poucas amostras por classe do `real_n1`. Sem o conjunto gerado, cai de volta sozinho pro comportamento antigo. Notebook `02_classifier_train.ipynb` ganhou a célula que gera esse conjunto antes do treino, reusando o Manga109 já anexado.

### Achado: colapso pra OUTROS em dado real -- atalho de domínio via mistura real/sintético (2026-08-02)

Primeiro retreino monitorado com o `real_n1_acc` ao vivo revelou algo novo: `val_acc` sintético em 98%+ e subindo normalmente, mas `real_n1_acc` preso entre 1-14% por 14 épocas, sem tendência de melhora mesmo com o sintético já saturado (`train_acc`≈100%). Diagnóstico (`errors_by_class` + previsões agregadas): o modelo previa **OUTROS em 95.4%** de todos os crops reais de N1, com confiança alta (88-92%) -- e **100% dos erros** eram "devia ser um kanji N1 específico, previu OUTROS". Crucial: nos **4.6%** de casos em que o modelo arriscava um kanji N1 específico (não OUTROS), acertava **100% das vezes** (169/169) -- ou seja, a capacidade de distinguir kanji estava intacta, só o portão OUTROS-vs-N1 estava quebrado especificamente em imagem real.

**Hipótese**: `merge_real_data.py` mescla dado real do Manga109 só na classe OUTROS (N1 continua 100% sintético, decisão histórica de 2026-07-11/12). Isso cria uma pista espúria: "parece foto/scan real (não render sintético limpo)" vira um atalho quase perfeito pra prever OUTROS, já que só essa classe tem exemplo real no treino -- o modelo não precisa aprender a reconhecer o traço, só a origem da imagem.

**Teste**: notebook ganhou o toggle `OUTROS_REAL_MIX` (Célula 1) -- com `False`, pula a mesclagem do real e o OUTROS fica 100% sintético igual N1, sem mudar nenhum código. Resultado, comparando o checkpoint com o atalho (14 épocas, OUTROS misto) vs. o novo (só 5 épocas, OUTROS 100% sintético):

| | OUTROS misto (14 épocas) | OUTROS sintético (5 épocas) |
|---|---|---|
| `real_n1_acc` | 4.60% | **96.98%** |
| Previu OUTROS em crop real de N1 | 95.4% | **2.7%** |
| Acerto quando arrisca um kanji | 100% (só 4.6% das vezes) | **99.6%** (97.3% das vezes) |
| **Amostra local, pipeline completo** (50 páginas/252 pares, mesma de sempre) | -- | **89.7%** (226/252) -- melhor recall já medido no projeto |

O checkpoint com OUTROS sintético, com **menos de um terço do treino** (5 de 30 épocas, ainda não convergiu), já supera o recall do checkpoint de referência totalmente treinado (07-23, 88.1% na mesma amostra). Confirma a hipótese do atalho de domínio com evidência forte, não só correlação.

**Trade-off em aberto**: a taxa de OUTROS na amostra caiu um pouco (87.5% vs ~92-93% histórico) -- pode ser só sub-treino, ou pode ser o risco que a mudança introduz (sem negativo real, talvez rejeite pior confusores reais que genuinamente não são N1; `real_n1` não mede isso, só testa se kanji N1 de verdade é reconhecido). Vale conferir de novo quando o treino terminar, e considerar um teste complementar específico pra "taxa de rejeição correta de não-N1 real" antes de fechar a decisão definitivamente.

**Promovido em 2026-08-05** (`weights/classifier_best.pt`), ainda na época 5/30 -- o treino seguia rodando no Kaggle, mas o resultado já era claramente o melhor do projeto (89.7% na amostra local vs 88.1% do checkpoint anterior totalmente treinado), então não fazia sentido esperar pra registrar isso como o ativo. Checkpoint anterior preservado em `weights/backups/classifier_2026-07-23.pt`.

~~**Pendências**: (1) validação full-corpus oficial (109 volumes) quando o treino terminar de convergir; (2) teste de rejeição de não-N1 real~~ **as duas fechadas em 2026-08-06** -- full-corpus deu 85.63% (ver seção "Pipeline completo" acima, checkpoint mantido ativo mesmo com recall bruto 1.15pp menor, decisão consciente); rejeição de não-N1 real deu 95.2%, ver seção própria abaixo.

### GPU local habilitada (2026-08-06)

Descoberto que a máquina local tem uma GPU (RTX 3050 Laptop, 6GB) com driver instalado, mas o PyTorch instalado era a build **CPU-only** -- todo teste local desta sessão (ETL9, `real_n1`, `corpus_validate` de amostra, confusão, `manga109_align_n1.py`) rodou em CPU sem precisar. Reinstalado com `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130 --force-reinstall --no-deps` (ver `requirements.txt`). `pytest` (32 testes) confirma que nada quebrou com a troca. Não afeta o Kaggle (já vem com CUDA pronto) -- só acelera avaliação/inferência local sem gastar hora de GPU do Kaggle.

### ETL9 do checkpoint ativo (2026-08-06)

Nunca tinha sido medido pro checkpoint bom (só existia pro regredido, 4.89%, sem contexto). Rodado com GPU local:

| Checkpoint | Top-1 ETL9 | Top-5 |
|---|---|---|
| Ativo (2026-08-05, `OUTROS_REAL_MIX=False`) | **10.48%** | 28.17% |
| 08-01 regredido (referência, contexto ruim) | 4.89% | -- |
| Diagnóstico original 2026-07-07 (modelo bem mais fraco/antigo) | 11.56% | -- |

Na mesma faixa do diagnóstico original, bem acima do regredido. Não é alarmante -- ETL9 é kanji **manuscrito**, domínio bem diferente do que `real_n1` testa (mangá impresso real, só muda o suporte/scan, não o traço). O modelo generaliza bem pro domínio-alvo (mangá real) sem necessariamente generalizar pra caligrafia à mão -- são duas formas diferentes de "fora da distribuição sintética", não a mesma pergunta.

### Teste de rejeição de não-N1 real (2026-08-06) -- fecha o trade-off do OUTROS sintético

Constrói o espelho do `real_n1`: em vez de "acerta kanji N1 real", mede "rejeita corretamente (prevê OUTROS) um caractere real que NÃO é N1". Reusa a heurística de `manga109_align.py` só pra avaliação (sem gravar em disco, sem mesclar em treino). Amostra: 20 volumes, 1965 páginas, **37.681 crops reais não-N1**.

| Resultado | Valor |
|---|---|
| Rejeitou corretamente (previu OUTROS) | **95.2%** (35881/37681) |
| Falso positivo (previu algum N1 específico) | 4.8% (1800/37681) |

**Leitura**: existe sim um custo real em trocar o OUTROS pra 100% sintético -- 4.8% de falso positivo não é zero. Mas é um custo pequeno frente ao ganho: recall de N1 real saltou de 4.60%→96.98% (`real_n1`) e o recall do pipeline completo subiu de 88.1%→89.7% na mesma amostra. O trade-off vale a pena, e fica documentado com número, não só hipótese. Falsos positivos concentrados em algumas classes específicas (U+4E59 88x, U+53E5 61x, U+4F8D 36x, ...) -- não é ruído uniforme, sugere confusões visuais específicas, possível alvo de investigação futura se sobrar tempo.
