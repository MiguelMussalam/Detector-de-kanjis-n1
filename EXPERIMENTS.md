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
| 2026-07-18 (rodada corrigida) | `weights/best.pt` (ativo) | ~17 imagens reais (Roboflow) + 300 páginas sintéticas (`generate_pages.py`, grade 2D multi-coluna em `dividir_em_celulas`), confirmadamente fundidas desta vez (~38 batches/epoch, batch=8 -> ~300 imgs/epoch, condizente com o dataset completo). `val` 100% real. 150 epochs, `yolo26n.pt`, imgsz 1024, batch 8. | **mAP@50 0.81** (epoch 146 pico 0.813, epoch 150 final 0.810), mAP@50-95 ~0.51, precision ~0.90, recall ~0.78-0.79 | Corrigiu o bug de resolução de diretório do Manga109 no Kaggle (`_buscar_manga109_diretorio` escolhia o dataset Roboflow por engano — ver `config.py`/`generate_pages.py`) que fez a rodada anterior (abaixo) rodar com bem menos dado do que deveria. `detector_fp_check.py` (150 páginas sem `<text>`, checkpoint anterior vs este): **145.9→22.8 detecções/página, 94.7%→22.0% páginas com detecção** — queda grande, mas coerente com recall/precision TAMBÉM subindo no conjunto rotulado (não é um detector menos sensível de forma geral, parece estar discriminando melhor traço-de-arte vs caractere). Checkpoint anterior preservado em `weights/backups/detector_2026-07-18.pt`. |
| 2026-07-18 (rodada com bug, superada) | `weights/backups/detector_2026-07-18.pt` | ~17 imagens reais + tentativa de 300 páginas sintéticas — **mas o número de batches/epoch observado (~3.75, contra ~38 da rodada corrigida) sugere fortemente que o dado sintético NÃO foi incluído de verdade nesta rodada** (mesmo bug de resolução de diretório, não percebido na hora porque essa rodada não veio com o log de geração). Não dá pra confirmar com certeza retroativamente (log da geração não foi salvo). | mAP@50 0.72-0.74 | Superada pela rodada corrigida acima. Ganho sobre o baseline de 2026-07-10 provavelmente veio majoritariamente de hiperparâmetro/versão do YOLO (`yolo26n.pt`, imgsz 1024), não do dado sintético — registrado aqui só como hipótese, não fato confirmado. |
| 2026-07-10 | `weights/backups/detector_2026-07-10.pt` | ~17 imagens reais anotadas via Roboflow (`miguelmussalam/manga109-character-bouding-box`), sem nenhum dado sintético | mAP@50 ~0.64 (baseline) | Baseline original. Superado pelas duas rodadas acima. |
