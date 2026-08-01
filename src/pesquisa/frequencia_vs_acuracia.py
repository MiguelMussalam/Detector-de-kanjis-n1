"""
frequencia_vs_acuracia.py
==========================
Cruza dois dados que já existiam separados: a frequência real de cada kanji
N1 no corpus Manga109 (`src/pesquisa/kanji_coverage.py`) e a acurácia
individual do classificador por classe no val sintético
(`src/classifier/eval.py --only confusao`).

Pergunta: o classificador erra mais nos kanji que são raros no corpus real,
ou o treino 100% sintético (mesma quantidade de amostra por classe,
independente da raridade real) compensa igual pra todo mundo?

Isso não estava respondido antes -- os dois CSVs existiam, mas nunca tinham
sido cruzados.

Uso:
    python -m src.pesquisa.frequencia_vs_acuracia
"""

import csv
import os

import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from config import ROOT_DIR

ACURACIA_CSV = os.path.join(ROOT_DIR, "data", "classifier_eval", "acuracia_por_classe.csv")
COBERTURA_CSV = os.path.join(ROOT_DIR, "data", "pesquisa", "kanji_coverage", "cobertura_n1.csv")
OUT_DIR = os.path.join(ROOT_DIR, "data", "pesquisa", "frequencia_vs_acuracia")

_AZUL = "#2a78d6"
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


def _carregar_acuracia() -> dict:
    if not os.path.exists(ACURACIA_CSV):
        raise FileNotFoundError(
            f"{ACURACIA_CSV} não encontrado -- rode "
            "`python -m src.classifier.eval --only confusao` primeiro."
        )
    with open(ACURACIA_CSV, encoding="utf-8") as f:
        return {row["kanji"]: float(row["acuracia_pct"]) for row in csv.DictReader(f)}


def _carregar_cobertura() -> dict:
    if not os.path.exists(COBERTURA_CSV):
        raise FileNotFoundError(
            f"{COBERTURA_CSV} não encontrado -- rode "
            "`python -m src.pesquisa.kanji_coverage` primeiro."
        )
    with open(COBERTURA_CSV, encoding="utf-8") as f:
        return {row["kanji"]: int(row["ocorrencias_no_corpus"]) for row in csv.DictReader(f)}


def cruzar(out_dir: str = OUT_DIR) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    acuracia = _carregar_acuracia()
    cobertura = _carregar_cobertura()

    linhas = []
    for kanji, acc in acuracia.items():
        if kanji not in cobertura:
            continue
        linhas.append({"kanji": kanji, "ocorrencias_no_corpus": cobertura[kanji], "acuracia_pct": acc})

    csv_path = os.path.join(out_dir, "frequencia_vs_acuracia.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["kanji", "ocorrencias_no_corpus", "acuracia_pct"])
        w.writeheader()
        w.writerows(linhas)

    # Correlacao de Spearman (rank, nao Pearson -- distribuicao de ocorrencias
    # e muito assimetrica, spearman nao assume linear nem normal).
    ocorrencias = [l["ocorrencias_no_corpus"] for l in linhas]
    acuracias = [l["acuracia_pct"] for l in linhas]
    rho, pvalor = spearmanr(ocorrencias, acuracias)

    faixas_def = [
        ("0", lambda n: n == 0),
        ("1-3", lambda n: 1 <= n <= 3),
        ("4-10", lambda n: 4 <= n <= 10),
        ("11-50", lambda n: 11 <= n <= 50),
        ("51+", lambda n: n >= 51),
    ]
    stats_faixas = []
    for nome, teste in faixas_def:
        subset = [l["acuracia_pct"] for l in linhas if teste(l["ocorrencias_no_corpus"])]
        if subset:
            stats_faixas.append({"faixa": nome, "n_classes": len(subset),
                                "acuracia_media": round(sum(subset) / len(subset), 1)})

    fig, ax = plt.subplots(figsize=(7.5, 4.5), facecolor=_SUPERFICIE)
    _estilo_eixo(ax)
    nomes = [s["faixa"] for s in stats_faixas]
    medias = [s["acuracia_media"] for s in stats_faixas]
    bars = ax.bar(nomes, medias, color=_AZUL, width=0.6, zorder=3)
    for bar, s in zip(bars, stats_faixas):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{s['acuracia_media']:.0f}%\n(n={s['n_classes']})",
                ha="center", va="bottom", fontsize=9.5, color=_TINTA)
    ax.set_ylim(0, 112)
    ax.set_ylabel("Acurácia média das classes (val sintético)", fontsize=11, color=_TINTA_SECUNDARIA)
    ax.set_xlabel("Ocorrências do kanji no corpus Manga109 (raridade real)", fontsize=11, color=_TINTA_SECUNDARIA)
    ax.set_title("Acurácia do classificador por faixa de raridade real do kanji",
                fontsize=13.5, color=_TINTA, fontweight="bold", pad=28, loc="left")
    ax.text(0.0, 1.05, f"correlação de Spearman: ρ={rho:.3f} (p={pvalor:.3f})",
            transform=ax.transAxes, fontsize=10, color=_TINTA_MUTED, ha="left")
    fig.tight_layout()
    png_path = os.path.join(out_dir, "frequencia_vs_acuracia.png")
    fig.savefig(png_path, dpi=150, facecolor=_SUPERFICIE)
    plt.close(fig)

    print(f"[INFO] {len(linhas)} classes cruzadas")
    print(f"[INFO] Correlacao de Spearman: rho={rho:.3f}, p-valor={pvalor:.3f}")
    for s in stats_faixas:
        print(f"  {s['faixa']:>6}: acc media {s['acuracia_media']:5.1f}%  (n={s['n_classes']})")
    print(f"[INFO] {csv_path}")
    print(f"[INFO] {png_path}")

    return {"csv": csv_path, "png": png_path, "rho": rho, "pvalor": pvalor, "faixas": stats_faixas}


def main():
    cruzar()


if __name__ == "__main__":
    main()
