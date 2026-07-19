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
| 2026-07-18 | `weights/best.pt` (ativo) | ~17 imagens reais (Roboflow) + 300 páginas sintéticas (`generate_pages.py`, grade 2D multi-coluna em `dividir_em_celulas` — ver `src/helper/measure_grid_rescue.py`: ~97% das linhas `<text>` do corpus viram aproveitáveis, contra ~25% antes da grade). `val` 100% real. 150 epochs, `yolo26n.pt`, imgsz 1024, batch 8. | **mAP@50 0.72-0.74** (epoch 131 pico 0.738, epoch 150 final 0.721), mAP@50-95 ~0.40-0.41, precision ~0.82-0.84, recall ~0.69 | Melhoria sobre o baseline de 2026-07-10 (mAP@50 ~0.64). `detector_fp_check.py` (150 páginas sem `<text>`, antigo vs novo): 150.9→145.9 detecções/página, 94.0%→94.7% páginas com detecção — **sem sinal de que o sintético ensinou "detectar qualquer coisa"**. Checkpoint antigo preservado em `weights/backups/detector_2026-07-10.pt`. |
| 2026-07-10 | `weights/backups/detector_2026-07-10.pt` | ~17 imagens reais anotadas via Roboflow (`miguelmussalam/manga109-character-bouding-box`), sem nenhum dado sintético | mAP@50 ~0.64 (baseline) | Superado pela rodada de 2026-07-18. Mantido como backup pra comparação. |
