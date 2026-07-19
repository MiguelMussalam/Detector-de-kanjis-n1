# Log de Experimentos

Registro das rodadas de treino do projeto (detector + classificador), pra reconstruir o histórico depois (útil pro relatório de IC). Este log **começa em 2026-07-16**, quando foi criado — só tem entradas verificadas com confiança nesta sessão. Rodadas anteriores não entraram aqui pra não arriscar registrar hiperparâmetro/número de memória de forma imprecisa; ver `weights/backups/manifest.md` para o que se sabe com certeza sobre os checkpoints antigos preservados (só data e contexto factual, sem detalhe de treino não confirmado).

Convenção: preencher uma linha nova a cada rodada de treino completa (não a cada ajuste de config testado localmente).

## Classificador

| Data | Checkpoint | Dado de treino | Resultado | Notas |
|---|---|---|---|---|
| 2026-07-12 | `weights/classifier_best.pt` (ativo) | 100% sintético — N1 200 amostras/classe (1232 classes), OUTROS 20000 treino / 5000 val, sem nenhum dado real misturado (`CLASSIFIER_SAMPLES_TRAIN=200`, `CLF_OUTROS_SAMPLES_TRAIN=20000`) | Recall no benchmark de região (Manga109, oficial `<text>`): **82.6%** (1 página) / **64.0%** (4 páginas curadas manualmente) / **78.0%** (corpus completo, 8536 páginas com N1, 109 volumes, 42611 pares esperados) | Resolveu o colapso de atalho de domínio (real vs. sintético) de rodadas anteriores ao remover dado real do treino inteiramente. **Ainda carrega o problema de erosão** (`CLF_MORFO_K_MAX=3` na época) corrigido só depois deste treino (ver abaixo) — não foi retreinado com a correção ainda. |

### Pendências conhecidas não incorporadas neste checkpoint
- `CLF_MORFO_K_MAX` corrigido de 3→2 em 2026-07-16 (auditoria de filtros mostrou penhasco de acurácia 92.5%→40% entre k=2 e k=3) — **não retreinado ainda**.
- Fração de amostras "tofu box" (fonte sem glifo pro caractere sorteado, ~12% num lote de 200 sanity) — investigado, causa entendida, sem correção aplicada ainda.

## Detector

| Data | Checkpoint | Dado de treino | Resultado | Notas |
|---|---|---|---|---|
| 2026-07-10 | `weights/best.pt` (ativo) | ~17 imagens reais anotadas via Roboflow (`miguelmussalam/manga109-character-bouding-box`), sem nenhum dado sintético | mAP@50 ~0.64 (baseline) | Dado sintético de página (`src/detector/generate_pages.py`, ancorado em bbox `<text>` oficial do Manga109) construído e auditado visualmente em lote pequeno, mas **nunca usado num treino de verdade** até o momento deste log. |

### Próxima rodada planejada (não executada ainda)
- Gerar dataset sintético completo do detector (`python -m src.detector.generate_pages`) e fundir no split de treino (`notebooks/01_detector_train.ipynb`, célula 4.5) — `val` continua 100% real.
- Comparar `src/helper/detector_fp_check.py` (taxa de detecção em páginas sem `<text>`) antes/depois, pra confirmar que o sintético não ensinou o detector a "detectar qualquer coisa".
