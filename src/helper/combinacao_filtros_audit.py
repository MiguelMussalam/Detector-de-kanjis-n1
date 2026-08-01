"""
combinacao_filtros_audit.py
=============================
Audita o efeito COMBINADO das degradações do classificador -- ao contrário
de `filter_audit.py` (que testa cada uma isolada, com as outras desligadas),
aqui a amostra é gerada via `generate_sample()` de verdade, com a mesma
aleatoriedade e as mesmas probabilidades da produção.

Contexto: cada degradação isolada já se mostrou segura (`filter_audit.py`,
97.5%+ de acurácia até e além do limite de produção). A hipótese testada
aqui é que o problema real não é nenhum filtro isolado no limite -- é a
COMBINAÇÃO, que já acontece o tempo todo (soma das probabilidades das 8
degradações gated ≈ 4.4 disparando juntas por amostra, em média), e que o
`filter_audit.py` não consegue ver por desenho (sempre desliga os outros
filtros).

O docstring de `generate_sample()` (src/classifier/generate_crops.py) cita
uma taxa de "~10-12%" de amostras fora da faixa de legibilidade -- número
de uma calibração pontual de 200 amostras (ver EXPERIMENTS.md:22, antes de
vários ajustes de config), nunca remedido em escala nem cruzado com "quais
filtros dispararam junto". Este script substitui essa cifra por uma medida
de verdade, e testa diretamente se a QUANTIDADE de degradações disparando
ao mesmo tempo (não a severidade de uma isolada) explica os piores casos.

Dois sinais de falha, cruzados (não fundidos num só, pra não ficar cego pra
amostra que o classificador erra com confiança sobre entrada ilegível, nem
pra amostra ilegível que o classificador "acerta" por sorte):
  - sinal do classificador: previsão errada ou OUTROS ou confiança baixa;
  - sinal independente do modelo: `legivel_final` (mesmo critério de
    contraste/fill_ratio que `generate_sample()` já usa internamente pra
    decidir se a amostra ficou dentro da faixa esperada).

Uso:
    python -m src.helper.combinacao_filtros_audit --n-amostras 2000
"""

import argparse
import csv
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from config import (
    ROOT_DIR, CLASSIFIER_INPUT_SIZE, CLASSIFIER_CANVAS_MARGIN,
    CLASSIFIER_FONT_SIZES, CLASSIFIER_FONT_SIZE_WEIGHTS,
    CLF_WEIGHTS_PATH, CLF_OUTROS_LABEL, PIPELINE_CLS_CONF_LOW,
)
from src.classifier.dataset import build_transform
from src.classifier.generate_crops import generate_sample
from src.helper.fonts import get_fonts_list
from src.helper.grade_visual import salvar_grade_visual
from src.helper.kanjis import get_kanjis
from src.pipeline.inference import load_classifier

OUT_DIR = os.path.join(ROOT_DIR, "data", "benchmark", "combinado")

# As 8 degradacoes com probabilidade propria (fundo nao entra na contagem de
# concorrencia -- sempre aplica, nao e' pass/fail, so' escolhe qual pool).
_DEGRADACOES = {
    "morfologia_k": "morfologia",
    "rotacao_graus": "rotacao",
    "translacao_dx": "translacao",
    "blur_sigma": "blur",
    "ruido_std": "ruido",
    "brilho_delta": "brilho",
    "contraste_fator": "contraste",
    "jpeg_qualidade": "jpeg",
}

_AZUL = "#2a78d6"
_HANKO = "#a8362e"
_TINTA = "#0b0b0b"
_TINTA_SECUNDARIA = "#52514e"
_TINTA_MUTED = "#898781"
_GRADE = "#e1e0d9"
_SUPERFICIE = "#fcfcfb"


def _estilo_eixo(ax):
    ax.set_facecolor(_SUPERFICIE)
    for lado in ("top", "right", "left"):
        ax.spines[lado].set_visible(False)
    ax.spines["bottom"].set_color(_TINTA_MUTED)
    ax.tick_params(colors=_TINTA_SECUNDARIA, labelsize=10)
    ax.yaxis.grid(True, color=_GRADE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def _concorrencia(log: dict) -> int:
    """Quantas das 8 degradações gated dispararam nessa amostra (0-8; na
    prática o piso é 1, já que blur tem probabilidade 1.0 e nunca sorteia
    sigma zero quando dispara -- não é bug, é o comportamento esperado)."""
    return sum(1 for chave in _DEGRADACOES if log.get(chave) is not None)


def _resumo_filtros(linha: dict) -> str:
    """Lista curta de quais degradacoes dispararam + severidade, pra legenda da grade visual."""
    partes = []
    if linha["morfologia_k"] is not None:
        partes.append(f"morf k={linha['morfologia_k']}({linha['morfologia_op']})")
    if linha["rotacao_graus"] is not None:
        partes.append(f"rot {linha['rotacao_graus']:.1f}°")
    if linha["translacao_dx"] is not None:
        partes.append(f"transl ({linha['translacao_dx']},{linha['translacao_dy']})")
    if linha["blur_sigma"] is not None:
        partes.append(f"blur σ={linha['blur_sigma']:.2f}")
    if linha["ruido_std"] is not None:
        partes.append(f"ruído σ={linha['ruido_std']:.1f}")
    if linha["brilho_delta"] is not None:
        partes.append(f"brilho {linha['brilho_delta']:+.0f}")
    if linha["contraste_fator"] is not None:
        partes.append(f"contr {linha['contraste_fator']:.2f}x")
    if linha["jpeg_qualidade"] is not None:
        partes.append(f"jpeg q={linha['jpeg_qualidade']}")
    return ", ".join(partes) if partes else "(nenhuma disparou)"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-amostras", type=int, default=2000,
                        help="Quantas amostras gerar via generate_sample() real.")
    parser.add_argument("--n-grid", type=int, default=24,
                        help="Quantos dos piores casos mostrar na grade visual.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    n1 = get_kanjis("n1")
    fonts = get_fonts_list()
    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    print("[INFO] Carregando classificador...")
    model, classes = load_classifier(CLF_WEIGHTS_PATH, device="cpu")
    model.eval()
    transform = build_transform()
    idx_outros = classes.index(CLF_OUTROS_LABEL)

    os.makedirs(OUT_DIR, exist_ok=True)
    linhas = []

    print(f"[INFO] Gerando {args.n_amostras} amostras via generate_sample() real...")
    for i in range(args.n_amostras):
        char = rng.choice(n1)
        font_path = rng.choice(fonts)
        font_size = rng.choices(CLASSIFIER_FONT_SIZES, weights=CLASSIFIER_FONT_SIZE_WEIGHTS)[0]

        log = {}
        img = generate_sample(
            char=char, font_path=font_path, font_size=font_size, rng=rng,
            output_size=CLASSIFIER_INPUT_SIZE, canvas_margin=CLASSIFIER_CANVAS_MARGIN,
            fonts_fallback=fonts, log=log,
        )

        with torch.no_grad():
            x = transform(Image.fromarray(img)).unsqueeze(0)
            probs = torch.softmax(model(x), dim=1)[0]
            pred_idx = int(probs.argmax())
            conf = float(probs[pred_idx])
        pred_classe = classes[pred_idx]
        correto = (pred_classe != CLF_OUTROS_LABEL and pred_classe == f"U+{ord(char):04X}")
        outros = pred_idx == idx_outros
        falha_classificador = (not correto) or conf < PIPELINE_CLS_CONF_LOW

        linha = {
            "char": char, "fonte": os.path.basename(font_path), "tamanho": font_size,
            "concorrencia": _concorrencia(log),
            "legivel_final": bool(log.get("legivel_final")),
            "tentativas_estagio2": log.get("tentativas_estagio2"),
            "usou_fallback_sem_blur_ruido": bool(log.get("usou_fallback_sem_blur_ruido")),
            "correto": correto, "outros": outros, "conf": round(conf, 3),
            "falha_classificador": falha_classificador,
        }
        for chave in _DEGRADACOES:
            linha[chave] = log.get(chave)
        linha["morfologia_op"] = log.get("morfologia_op")
        linha["translacao_dy"] = log.get("translacao_dy")
        linha["fundo_escuro"] = log.get("fundo_escuro")
        linha["_img"] = img
        linhas.append(linha)

    # --- CSV (sem a imagem) ---
    csv_path = os.path.join(OUT_DIR, "amostras.csv")
    campos_csv = [k for k in linhas[0].keys() if k != "_img"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=campos_csv)
        w.writeheader()
        for linha in linhas:
            w.writerow({k: linha[k] for k in campos_csv})
    print(f"[INFO] {csv_path}")

    # --- Taxa de aviso medida fresca, em escala (substitui o "10-12%" antigo) ---
    n_total = len(linhas)
    n_ilegivel = sum(1 for l in linhas if not l["legivel_final"])
    n_falha_clf = sum(1 for l in linhas if l["falha_classificador"])
    pct_ilegivel = 100 * n_ilegivel / n_total
    pct_falha_clf = 100 * n_falha_clf / n_total
    print(f"\n[RESUMO] Amostras: {n_total}")
    print(f"[RESUMO] Ilegível (legivel_final=False, sinal independente do modelo): "
          f"{n_ilegivel} ({pct_ilegivel:.1f}%)")
    print(f"[RESUMO] Falha do classificador (errou/OUTROS/confiança<{PIPELINE_CLS_CONF_LOW}): "
          f"{n_falha_clf} ({pct_falha_clf:.1f}%)")

    # --- Concorrência vs acurácia ---
    concorrencias = sorted(set(l["concorrencia"] for l in linhas))
    stats_concorrencia = []
    for c in concorrencias:
        subset = [l for l in linhas if l["concorrencia"] == c]
        acc = 100 * sum(l["correto"] for l in subset) / len(subset)
        pct_outros = 100 * sum(l["outros"] for l in subset) / len(subset)
        conf_media = sum(l["conf"] for l in subset) / len(subset)
        stats_concorrencia.append({
            "concorrencia": c, "n": len(subset), "acuracia_pct": round(acc, 1),
            "pct_outros": round(pct_outros, 1), "conf_media": round(conf_media, 3),
        })

    print(f"\n{'concorrência':>12} {'n':>6} {'acurácia':>10} {'% OUTROS':>10} {'conf':>8}")
    for s in stats_concorrencia:
        print(f"{s['concorrencia']:>12} {s['n']:>6} {s['acuracia_pct']:>9.1f}% "
              f"{s['pct_outros']:>9.1f}% {s['conf_media']:>8.3f}")

    fig, ax = plt.subplots(figsize=(7.5, 4.5), facecolor=_SUPERFICIE)
    _estilo_eixo(ax)
    xs = [s["concorrencia"] for s in stats_concorrencia]
    accs = [s["acuracia_pct"] for s in stats_concorrencia]
    bars = ax.bar([str(x) for x in xs], accs, color=_AZUL, width=0.6, zorder=3)
    for bar, s in zip(bars, stats_concorrencia):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{s['acuracia_pct']:.0f}%\nn={s['n']}", ha="center", va="bottom",
                fontsize=9, color=_TINTA)
    ax.set_ylim(0, 112)
    ax.set_ylabel("Acurácia", fontsize=11, color=_TINTA_SECUNDARIA)
    ax.set_xlabel("Nº de degradações disparando na mesma amostra (concorrência)",
                  fontsize=11, color=_TINTA_SECUNDARIA)
    ax.set_title("Acurácia cai com o número de degradações combinadas?",
                fontsize=13.5, color=_TINTA, fontweight="bold", pad=14, loc="left")
    fig.tight_layout()
    png_concorrencia = os.path.join(OUT_DIR, "concorrencia_vs_acuracia.png")
    fig.savefig(png_concorrencia, dpi=150, facecolor=_SUPERFICIE)
    plt.close(fig)
    print(f"[INFO] {png_concorrencia}")

    # --- Enriquecimento: taxa de disparo entre falhas vs sucessos, por degradação ---
    falhas = [l for l in linhas if l["falha_classificador"]]
    sucessos = [l for l in linhas if not l["falha_classificador"]]
    enriquecimento = []
    for chave, nome in _DEGRADACOES.items():
        taxa_falha = 100 * sum(1 for l in falhas if l[chave] is not None) / max(1, len(falhas))
        taxa_sucesso = 100 * sum(1 for l in sucessos if l[chave] is not None) / max(1, len(sucessos))
        enriquecimento.append({
            "degradacao": nome,
            "taxa_disparo_em_falhas_pct": round(taxa_falha, 1),
            "taxa_disparo_em_sucessos_pct": round(taxa_sucesso, 1),
            "diferenca_pp": round(taxa_falha - taxa_sucesso, 1),
        })
    enriquecimento.sort(key=lambda r: -r["diferenca_pp"])

    print(f"\n{'degradacao':>12} {'disparo em falha':>18} {'disparo em sucesso':>20} {'diferenca':>11}")
    for r in enriquecimento:
        print(f"{r['degradacao']:>12} {r['taxa_disparo_em_falhas_pct']:>17.1f}% "
              f"{r['taxa_disparo_em_sucessos_pct']:>19.1f}% {r['diferenca_pp']:>+10.1f}pp")

    # --- Tabela 2x2: legivel_final x falha_classificador ---
    cruzamento = {
        "ilegivel_e_classificador_errou": sum(1 for l in linhas if not l["legivel_final"] and l["falha_classificador"]),
        "ilegivel_mas_classificador_acertou": sum(1 for l in linhas if not l["legivel_final"] and not l["falha_classificador"]),
        "legivel_mas_classificador_errou": sum(1 for l in linhas if l["legivel_final"] and l["falha_classificador"]),
        "legivel_e_classificador_acertou": sum(1 for l in linhas if l["legivel_final"] and not l["falha_classificador"]),
    }

    # --- resumo.md ---
    resumo_path = os.path.join(OUT_DIR, "resumo.md")
    with open(resumo_path, "w", encoding="utf-8") as f:
        f.write("# Auditoria de degradações combinadas -- resumo\n\n")
        f.write(f"Amostras geradas via `generate_sample()` real (não isolada): {n_total}\n\n")
        f.write(f"- Ilegível (`legivel_final=False`, sinal independente do modelo): "
                f"{n_ilegivel} ({pct_ilegivel:.1f}%) -- substitui a cifra antiga de \"~10-12%\" "
                f"(EXPERIMENTS.md, calibração de 200 amostras)\n")
        f.write(f"- Falha do classificador (errou/OUTROS/confiança<{PIPELINE_CLS_CONF_LOW}): "
                f"{n_falha_clf} ({pct_falha_clf:.1f}%)\n\n")

        f.write("## Concorrência (nº de degradações disparando juntas) vs. acurácia\n\n")
        f.write("Nota: o piso prático é concorrência=1 (blur tem probabilidade 1.0 e nunca "
                "sorteia severidade zero quando dispara) -- concorrência=0 fica estruturalmente "
                "vazio, não é um dado faltando.\n\n")
        f.write("| concorrência | n | acurácia | % OUTROS | confiança média |\n|---|---|---|---|---|\n")
        for s in stats_concorrencia:
            f.write(f"| {s['concorrencia']} | {s['n']} | {s['acuracia_pct']:.1f}% | "
                    f"{s['pct_outros']:.1f}% | {s['conf_media']:.3f} |\n")

        f.write("\n## Enriquecimento: quais degradações aparecem mais nas falhas\n\n")
        f.write("| degradação | disparo em falhas | disparo em sucessos | diferença |\n|---|---|---|---|\n")
        for r in enriquecimento:
            f.write(f"| {r['degradacao']} | {r['taxa_disparo_em_falhas_pct']:.1f}% | "
                    f"{r['taxa_disparo_em_sucessos_pct']:.1f}% | {r['diferenca_pp']:+.1f}pp |\n")

        f.write("\n## Legibilidade (sinal independente) x acerto do classificador\n\n")
        f.write("| | classificador errou/OUTROS/baixa confiança | classificador acertou |\n")
        f.write("|---|---|---|\n")
        f.write(f"| **ilegível** (`legivel_final=False`) | {cruzamento['ilegivel_e_classificador_errou']} "
                f"(falha inequívoca) | {cruzamento['ilegivel_mas_classificador_acertou']} "
                f"(modelo tolerou entrada ruim) |\n")
        f.write(f"| **legível** (`legivel_final=True`) | {cruzamento['legivel_mas_classificador_errou']} "
                f"(fraqueza do classificador em entrada limpa) | {cruzamento['legivel_e_classificador_acertou']} "
                f"(ok) |\n")
    print(f"[INFO] {resumo_path}")

    # --- Grade visual dos piores casos ---
    if args.n_grid > 0:
        piores = sorted(
            linhas,
            key=lambda l: (l["legivel_final"], not l["falha_classificador"], l["conf"]),
        )[:args.n_grid]
        itens = [{
            "imagem": l["_img"],
            "titulo": (f"{l['char']} | conf={l['conf']:.2f} "
                       f"{'OUTROS' if l['outros'] else ('ERRO' if not l['correto'] else 'ok')}\n"
                       f"{'ILEGÍVEL' if not l['legivel_final'] else 'legível'} | "
                       f"concorrência={l['concorrencia']}\n{_resumo_filtros(l)}"),
            "cor": "red" if l["falha_classificador"] or not l["legivel_final"] else "green",
            "cmap": "gray",
        } for l in piores]
        png_grid = os.path.join(OUT_DIR, "grade_piores.png")
        salvar_grade_visual(itens, png_grid, n=len(itens), seed=args.seed, cols=6, figsize_cel=(2.6, 3.0))


if __name__ == "__main__":
    main()
