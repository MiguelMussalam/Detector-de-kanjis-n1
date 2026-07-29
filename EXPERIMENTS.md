# Log de Experimentos

Registro das rodadas de treino do projeto (detector + classificador), pra reconstruir o histórico depois (útil pro relatório de IC). Este log **começa em 2026-07-16**, quando foi criado — só tem entradas verificadas com confiança nesta sessão. Rodadas anteriores não entraram aqui pra não arriscar registrar hiperparâmetro/número de memória de forma imprecisa; ver `weights/backups/manifest.md` para o que se sabe com certeza sobre os checkpoints antigos preservados (só data e contexto factual, sem detalhe de treino não confirmado).

Convenção: preencher uma linha nova a cada rodada de treino completa (não a cada ajuste de config testado localmente).

## Classificador

| Data | Checkpoint | Dado de treino | Resultado | Notas |
|---|---|---|---|---|
| 2026-07-23 | `weights/classifier_best.pt` (ativo) | Mesma receita de 2026-07-19 + `CLF_STEM_LEVE=True` (stride do conv1 1→ e maxpool→Identity, ver `model.py`) + `CLF_LABEL_SMOOTHING=0.1`. Treino interrompido manualmente na época 15/30 (tempo por época ~5x maior, ver nota de custo abaixo) — patience=7 nunca disparou, `val_acc` ainda subindo (98.53% na época 15, já acima do 97.7% final da rodada sem stem_leve). | Benchmark leve (1 volume, 61 páginas, 207 pares esperados, métrica multiset já corrigida): **72.5%→75.4% (+2.9pp)** vs. checkpoint anterior — amostra pequena, sinal direcional, não conclusivo. | **Bug real encontrado e corrigido durante a validação**: o checkpoint não guardava se foi treinado com `stem_leve` (stride/Identity não aparecem no `state_dict`, só pesos) — carregar em outra máquina (`CLF_STEM_LEVE` local default `False`) reconstruía a arquitetura ERRADA, pesos carregavam sem erro de shape mas a rede saía incoerente (recall caiu pra **0%** no primeiro teste). Corrigido: checkpoint agora salva `"stem_leve"` no dict (`train.py`), `load_classifier` (`inference.py`) lê isso em vez de confiar no config local. Checkpoint já baixado foi corrigido manualmente (chave adicionada via script). **Custo de inferência real**: ~4.9x mais lento por chamada do classificador (2.61ms→12.75ms, CPU, medido direto), ~3.7x mais lento por página no pipeline completo — puxa mAIS resolução espacial (mapa 8x8 em vez de 2x2 antes do avgpool), mas com preço de latência nada desprezível pra um pipeline que chama o classificador uma vez por caractere detectado. Checkpoint anterior preservado em `weights/backups/classifier_2026-07-19.pt`. |
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
| 2026-07-25 | `best.pt` (mAP@50 0.81) | `classifier_best.pt` com `stem_leve` | **multiset (corrigida)** | **86.78%** (37789/43547) | 92.3% | 83.5 min (0.587s/pág, Kaggle GPU) |
| 2026-07-19 | `best.pt` (mAP@50 0.81) | `classifier_2026-07-19.pt` (sem stem_leve) | set deduplicado (antiga) | 82.69% (35234/42611) | 88.1% | 93.5 min |
| 2026-07-15 | `detector_2026-07-18.pt`/anterior (mAP@50 ~0.64-0.74) | mesmo de cima | set deduplicado (antiga) | 77.99% (33233/42611) | 88.0% | ~92 min |

**Nota metodológica**: a linha de 2026-07-25 muda métrica E checkpoint do classificador ao mesmo tempo em relação à linha de 2026-07-19 — não é uma comparação de variável única. Mas como o multiset é uma contagem mais **rigorosa** que o set deduplicado (nunca infla recall, só corrige a inflação anterior — ver nota de 2026-07-19 abaixo), ver o número subir mesmo assim (82.69%→86.78%) é um resultado mais forte do que uma leitura ingênua sugere, não mais fraco.

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
- Só testado com EasyOCR — Tesseract (`jpn_vert`) fica como comparação secundária, não feita ainda por restrição de tempo.
- **Lacuna identificada 2026-07-26**: a Hipótese 1 da proposta fala em recall de **detecção**, mas o teste atual isola reconhecimento (dá a bbox certa pro EasyOCR de graça). Falta um teste com o EasyOCR rodando detecção+reconhecimento próprios na página inteira (mesma amostra de 50 páginas), pra cobrir a hipótese como está literalmente escrita. Como o EasyOCR já perde feio mesmo com detecção de graça, a expectativa é que esse teste só amplie a diferença, não a reverta.
