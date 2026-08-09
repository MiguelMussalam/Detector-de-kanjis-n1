# Manifest de backups de checkpoints

Checkpoints antigos, renomeados por data real de modificação (não confiar em
nomes antigos tipo "antigo2"/"antigo3" — a ordem cronológica real não batia
com o nome, confirmado via `mtime` antes de renomear).

Os ativos (`weights/best.pt` detector, `weights/classifier_best.pt`
classificador) não estão aqui — continuam na raiz de `weights/`, são os
nomes que `config.py` (`DETECTOR_WEIGHTS_PATH`/`CLF_WEIGHTS_PATH`) espera por
padrão.

## Classificador

| Arquivo | Data | Contexto |
|---|---|---|
| `classifier_2026-07-09.pt` | 2026-07-09 | Checkpoint mais antigo preservado. Contexto de treino não reconstituível com certeza a partir desta conversa — só o fato/data ficou registrado. |
| `classifier_2026-07-11a.pt` | 2026-07-11 02:22 | Anterior ao `classifier_2026-07-11b.pt` no mesmo dia. Contexto de treino não reconstituível com certeza. |
| `classifier_2026-07-11b.pt` | 2026-07-11 19:58 | Imediatamente anterior ao checkpoint atual (instalado em 2026-07-12). Contexto de treino não reconstituível com certeza. |
| `classifier_2026-07-12.pt` | 2026-07-12 00:45 | **Rodada 100% sintética** (N1 200/classe + OUTROS 20000/5000 treino, sem nenhum dado real misturado) — resolveu o colapso de atalho de domínio de rodadas anteriores. Recall no benchmark: 82.6% (1 página) / 64.0% (4 páginas curadas) / **78.0% (corpus completo, 8536 páginas, 109 volumes)**. Ver `EXPERIMENTS.md` na raiz do repo. |
| `classifier_2026-07-23.pt` | 2026-07-23 | `stem_leve` + `label_smoothing=0.1`, treino interrompido na época 15/30, `val_acc=98.53%`. Recall corpus completo (multiset): **86.78%**, OUTROS 92.3%. Foi o ativo até 2026-08-05, quando foi substituído pelo checkpoint com `OUTROS_REAL_MIX=False` (ver linha abaixo). Ver `EXPERIMENTS.md`. |
| `classifier_2026-08-01_regressao.pt` | 2026-08-01 | Mesma receita de 2026-07-23 + fixes desta sessão (correção do no-op de `apply_translate_and_crop`, `CLF_MORFO_PROB_COM_TRANSLACAO` reduzindo co-ocorrência morfologia+translação, troca de 3 fontes por `GenEi-Antique-M`/`GenEi-Gothic-KL-H`). 18 epochs, `val_acc=98.82%` (sintético, melhor que o anterior). **Regressão confirmada em dado real**: amostra local de 50 páginas 84.5% (vs 88.1%), dry-run Kaggle de 3 volumes/216 páginas **71.9%** (919/1278, vs 86.78% do checkpoint anterior), ETL9 4.89% top-1. Causa raiz: bug no fix de translação fazia `apply_translate_and_crop` encolher o recorte em 100% das amostras (não só nas que translacionam de verdade), mudando o enquadramento do glifo de ~77% pra ~96% do frame sistematicamente. Foi ativo brevemente (2026-08-01), revertido no mesmo dia após a descoberta. **Preservado só pra referência/debug** -- não é candidato a reativação, o fix do crop_size (`generate_crops.py`) precisa de um retreino novo antes de tentar de novo. Ver `EXPERIMENTS.md`. |
| `classifier_2026-08-05_outros_sintetico.pt` | 2026-08-05 | Mesma receita de 2026-07-23 + fix do `crop_size` + `OUTROS_REAL_MIX=False`. Só época 5/30 (promovido com o treino ainda em andamento no Kaggle, `val_acc=97.87%` sintético). `real_n1_acc`=96.98%, rejeição de não-N1 real 95.2%. Full-corpus oficial: 85.63%. Foi o ativo de 2026-08-05 a 2026-08-08, substituído por `classifier_stem_convergido.pt` (linha abaixo) quando o treino continuou até convergir de verdade. Ver `EXPERIMENTS.md`. |
| `classifier_no_stem.pt` | 2026-08-08 | Mesma receita (`OUTROS_REAL_MIX=False`, fix do `crop_size`) mas `stem_leve=False`, treinado do zero (36 épocas, early stopping) -- experimento pra reconfirmar se `stem_leve` (decisão de 2026-07-25) ainda vale a pena com a receita atual. `real_n1_acc`=94.01%, rejeição 96.4%. **Perdeu pro `stem_leve=True` convergido** (linha abaixo) nas duas métricas -- preservado só como referência, não é candidato a promoção. Ver EXPERIMENTS.md "Experimento stem_leve=True vs False". |
| `weights/classifier_best.pt` (ativo, não está aqui) | 2026-08-08 | `classifier_stem_convergido.pt` -- continuação de `classifier_2026-08-05_outros_sintetico.pt` (`CLF_FINETUNE_FROM`) até convergir de verdade (11 épocas extras, early stopping, treinado localmente com GPU). **`real_n1_acc`=96.73%** (praticamente igual ao anterior, reconhecimento já saturado), mas **rejeição de não-N1 real subiu de 95.2%→97.5%** -- o treino extra melhorou a generalização do limite de decisão OUTROS-vs-N1 mesmo sem melhorar mais o reconhecimento em si. Confirma `stem_leve=True` como melhor escolha (vs `classifier_no_stem.pt` acima). Pendente: validação full-corpus oficial deste checkpoint especificamente (o 85.63% registrado é do checkpoint anterior, época 5). Ver `EXPERIMENTS.md`. |

## Detector

| Arquivo | Data | Contexto |
|---|---|---|
| `detector_2026-07-03.pt` | 2026-07-03 | Checkpoint mais antigo preservado. Contexto não reconstituível com certeza. |
| `detector_2026-07-09.pt` | 2026-07-09 | Imediatamente anterior ao ativo anterior. Contexto não reconstituível com certeza. |
| `detector_2026-07-10.pt` | 2026-07-10 | Treinado só com ~17 imagens reais anotadas via Roboflow (`miguelmussalam/manga109-character-bouding-box`), mAP@50 ~0.64. Foi o ativo até 2026-07-18. Ver `EXPERIMENTS.md`. |
| `detector_2026-07-18.pt` | 2026-07-18 | mAP@50 0.72-0.74. Foi o ativo até a rodada corrigida do mesmo dia — suspeita (não confirmada, log da geração não foi salvo) de que o dado sintético não entrou de fato no treino desta rodada, por causa do mesmo bug de resolução de diretório do Manga109 corrigido depois. Ver `EXPERIMENTS.md`. |
| `weights/best.pt` (ativo, não está aqui) | 2026-07-18 (rodada corrigida) | ~17 imagens reais + 300 páginas sintéticas (`src/detector/generate_pages.py`, grade multi-coluna), confirmadamente incluídas desta vez. mAP@50 0.81, recall/precision subiram junto com uma queda grande de detecção em página sem `<text>` (ver `detector_fp_check.py`) — sinal de melhor discriminação, não de detector menos sensível. Ver `EXPERIMENTS.md`. |

## Por que não apaguei nada

Preferi só reorganizar (mover + renomear) em vez de apagar os backups mais
antigos -- são poucos MB no total e não custam nada guardados, e servem de
histórico caso seja preciso comparar/reverter no futuro.
