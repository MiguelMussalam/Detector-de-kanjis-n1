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
| `weights/classifier_best.pt` (ativo, não está aqui) | 2026-07-12 00:45 | **Rodada 100% sintética** (N1 200/classe + OUTROS 20000/5000 treino, sem nenhum dado real misturado) — resolveu o colapso de atalho de domínio de rodadas anteriores. Recall no benchmark: 82.6% (1 página) / 64.0% (4 páginas curadas) / **78.0% (corpus completo, 8536 páginas, 109 volumes)**. Ver `EXPERIMENTS.md` na raiz do repo. |

## Detector

| Arquivo | Data | Contexto |
|---|---|---|
| `detector_2026-07-03.pt` | 2026-07-03 | Checkpoint mais antigo preservado. Contexto não reconstituível com certeza. |
| `detector_2026-07-09.pt` | 2026-07-09 | Imediatamente anterior ao ativo. Contexto não reconstituível com certeza. |
| `weights/best.pt` (ativo, não está aqui) | 2026-07-10 | Treinado só com ~17 imagens reais anotadas via Roboflow (`miguelmussalam/manga109-character-bouding-box`), mAP@50 ~0.64. Ainda **não** incorpora o dado sintético de página construído em `src/detector/generate_pages.py` (Parte B) — nunca foi usado num treino de verdade até o momento deste manifest. Ver `EXPERIMENTS.md`. |

## Por que não apaguei nada

Preferi só reorganizar (mover + renomear) em vez de apagar os backups mais
antigos -- são poucos MB no total e não custam nada guardados, e servem de
histórico caso seja preciso comparar/reverter no futuro.
