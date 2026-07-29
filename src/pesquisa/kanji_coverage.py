"""
kanji_coverage.py
==================
Mede quantas vezes os kanji de cada nível JLPT (N1-N5) aparecem de verdade
no corpus Manga109 -- frequência bruta de texto (conta ocorrência na
transcrição oficial, não depende de detecção nem alinhamento). Usado pra:

  1. Caracterizar a cobertura real dos 1232 kanji N1 (nosso alvo de
     classificação) no corpus: quantos nunca aparecem, quantos são raros,
     etc -- evidência de por que dado sintético é necessário mesmo com
     corpus grande (109 volumes).
  2. Comparar a frequência de aparição entre os níveis N1-N5 -- mostra
     quantitativamente que kanji "avançado" (N1) é mais raro no texto real
     do que kanji "básico" (N5), como o próprio design do JLPT prevê.

Primeiro script de `src/pesquisa/` -- pasta de scripts de OBTENÇÃO DE DADOS
(não geram nem avaliam modelo): cada um extrai um número/gráfico específico
do corpus real pra usar na apresentação da pesquisa.

Uso:
    python -m src.pesquisa.kanji_coverage
"""

import csv
import json
import os
from collections import Counter

import matplotlib.pyplot as plt

from config import ROOT_DIR
from src.helper.kanjis import get_kanjis
from src.helper.manga109_align import limpar_string
from src.helper.manga109_corpus import GT_PATH

OUT_DIR = os.path.join(ROOT_DIR, "data", "pesquisa", "kanji_coverage")

# Paleta validada (ver skill de dataviz do projeto) -- rampa ordinal azul,
# 5 degraus, N1 (mais claro) -> N5 (mais escuro): a cor reforça visualmente
# que N1..N5 é uma escala ordenada de raridade, não categorias soltas.
_RAMPA_ORDINAL_N1_N5 = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
_AZUL = "#2a78d6"
_TINTA = "#0b0b0b"
_TINTA_SECUNDARIA = "#52514e"
_TINTA_MUTED = "#898781"
_GRADE = "#e1e0d9"
_SUPERFICIE = "#fcfcfb"


def carregar_corpus() -> dict:
    if not os.path.exists(GT_PATH):
        raise FileNotFoundError(
            f"{GT_PATH} não encontrado -- rode `python -m src.helper.manga109_corpus` primeiro "
            "(reconstrói o ground truth completo a partir das anotações do Manga109)."
        )
    with open(GT_PATH, encoding="utf-8") as f:
        return json.load(f)


def contar_ocorrencias(kanji_set: set, gt: dict) -> Counter:
    """Conta ocorrências (multiset) de um conjunto de kanji na transcrição limpa do corpus inteiro."""
    contagem = Counter()
    for pagina in gt["paginas"]:
        for linha in pagina["linhas"]:
            for c in limpar_string(linha["transcricao"]):
                if c in kanji_set:
                    contagem[c] += 1
    return contagem


def _estilo_eixo(ax):
    ax.set_facecolor(_SUPERFICIE)
    for lado in ("top", "right", "left"):
        ax.spines[lado].set_visible(False)
    ax.spines["bottom"].set_color(_TINTA_MUTED)
    ax.tick_params(colors=_TINTA_SECUNDARIA, labelsize=10)
    ax.yaxis.grid(True, color=_GRADE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def cobertura_n1(gt: dict, out_dir: str = OUT_DIR) -> dict:
    """Cobertura dos 1232 kanji N1 no corpus -- CSV completo (1 linha/kanji) + gráfico de distribuição por faixa."""
    os.makedirs(out_dir, exist_ok=True)
    n1 = get_kanjis("n1")
    contagem = contar_ocorrencias(set(n1), gt)

    csv_path = os.path.join(out_dir, "cobertura_n1.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["kanji", "codepoint", "ocorrencias_no_corpus"])
        for k in sorted(n1, key=lambda c: -contagem.get(c, 0)):
            w.writerow([k, f"U+{ord(k):04X}", contagem.get(k, 0)])

    faixas_def = [
        ("0", lambda n: n == 0),
        ("1-3", lambda n: 1 <= n <= 3),
        ("4-10", lambda n: 4 <= n <= 10),
        ("11-50", lambda n: 11 <= n <= 50),
        ("51+", lambda n: n >= 51),
    ]
    faixas = [(nome, sum(1 for k in n1 if teste(contagem.get(k, 0)))) for nome, teste in faixas_def]

    fig, ax = plt.subplots(figsize=(7.5, 4.5), facecolor=_SUPERFICIE)
    _estilo_eixo(ax)
    nomes = [f[0] for f in faixas]
    valores = [f[1] for f in faixas]
    bars = ax.bar(nomes, valores, color=_AZUL, width=0.6, zorder=3)
    for bar, v in zip(bars, valores):
        ax.text(bar.get_x() + bar.get_width() / 2, v + max(valores) * 0.015, str(v),
                ha="center", va="bottom", fontsize=10.5, color=_TINTA)
    ax.set_ylim(0, max(valores) * 1.18)
    ax.set_ylabel("Classes N1", fontsize=11, color=_TINTA_SECUNDARIA)
    ax.set_xlabel("Ocorrências no corpus Manga109 (109 volumes)", fontsize=11, color=_TINTA_SECUNDARIA)
    ax.set_title("Cobertura real dos 1232 kanji N1 no Manga109", fontsize=14, color=_TINTA,
                fontweight="bold", pad=28, loc="left")
    n_zero = faixas[0][1]
    ax.text(0.0, 1.05, f"{n_zero} classes ({100 * n_zero / len(n1):.0f}%) nunca aparecem no corpus inteiro",
            transform=ax.transAxes, fontsize=10.5, color=_TINTA_MUTED, ha="left")
    fig.tight_layout()
    png_path = os.path.join(out_dir, "distribuicao_n1.png")
    fig.savefig(png_path, dpi=150, facecolor=_SUPERFICIE)
    plt.close(fig)

    print(f"[INFO] {csv_path}")
    print(f"[INFO] {png_path}")
    return {"csv": csv_path, "png": png_path, "faixas": faixas}


def comparar_niveis(gt: dict, out_dir: str = OUT_DIR) -> dict:
    """Compara frequência de aparição real no corpus entre kanji N1 (raro/avançado) e N5 (comum/básico)."""
    os.makedirs(out_dir, exist_ok=True)
    niveis = ["n1", "n2", "n3", "n4", "n5"]
    linhas = []
    for nivel in niveis:
        kanjis = get_kanjis(nivel)
        contagem = contar_ocorrencias(set(kanjis), gt)
        total = sum(contagem.values())
        n_zero = sum(1 for k in kanjis if contagem.get(k, 0) == 0)
        linhas.append({
            "nivel": nivel.upper(),
            "n_classes": len(kanjis),
            "ocorrencias_totais": total,
            "media_por_classe": round(total / len(kanjis), 1),
            "pct_zero": round(100 * n_zero / len(kanjis), 1),
        })

    csv_path = os.path.join(out_dir, "comparacao_niveis.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(linhas[0].keys()))
        w.writeheader()
        w.writerows(linhas)

    fig, ax = plt.subplots(figsize=(7.5, 4.5), facecolor=_SUPERFICIE)
    _estilo_eixo(ax)
    nomes = [l["nivel"] for l in linhas]
    medias = [l["media_por_classe"] for l in linhas]
    bars = ax.bar(nomes, medias, color=_RAMPA_ORDINAL_N1_N5, width=0.6, zorder=3)
    ax.set_yscale("log")
    for bar, l in zip(bars, linhas):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.08,
                f"{l['media_por_classe']:.0f}", ha="center", va="bottom", fontsize=10.5, color=_TINTA)
    ax.set_ylabel("Ocorrências médias por kanji (escala log)", fontsize=11, color=_TINTA_SECUNDARIA)
    ax.set_xlabel("Nível JLPT (N1 = mais avançado/raro  →  N5 = básico/comum)", fontsize=11, color=_TINTA_SECUNDARIA)
    ax.set_title("Frequência real de kanji por nível JLPT no Manga109", fontsize=14, color=_TINTA,
                fontweight="bold", pad=14, loc="left")
    fig.tight_layout()
    png_path = os.path.join(out_dir, "comparacao_niveis.png")
    fig.savefig(png_path, dpi=150, facecolor=_SUPERFICIE)
    plt.close(fig)

    print(f"[INFO] {csv_path}")
    print(f"[INFO] {png_path}")
    return {"csv": csv_path, "png": png_path, "linhas": linhas}


def main():
    gt = carregar_corpus()
    print(f"[INFO] Corpus carregado: {len(gt['paginas'])} páginas")
    cobertura_n1(gt)
    comparar_niveis(gt)


if __name__ == "__main__":
    main()
