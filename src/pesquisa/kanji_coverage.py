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

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from config import ROOT_DIR
from src.helper.fonts import get_fonts_list
from src.helper.kanjis import get_kanjis
from src.helper.manga109_align import limpar_string
from src.helper.manga109_corpus import GT_PATH

OUT_DIR = os.path.join(ROOT_DIR, "data", "pesquisa", "kanji_coverage")

# Paleta validada (ver skill de dataviz do projeto) -- rampa ordinal azul,
# 5 degraus, N1 (mais claro) -> N5 (mais escuro): a cor reforça visualmente
# que N1..N5 é uma escala ordenada de raridade, não categorias soltas. Usada
# nos graficos de BARRA (a altura ja mostra a ordem/magnitude).
_RAMPA_ORDINAL_N1_N5 = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]

# Paleta categorica validada (5 primeiros slots, ordem fixa) -- usada onde a
# tarefa e distinguir fatias/categorias a olho (pizza), nao mostrar magnitude
# por tom -- cores parecidas (mesma rampa) dificultam ver onde uma fatia
# termina e a outra comeca.
_CATEGORICA_5 = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
_AZUL = "#2a78d6"
_TINTA = "#0b0b0b"
_TINTA_SECUNDARIA = "#52514e"
_TINTA_MUTED = "#898781"
_GRADE = "#e1e0d9"
_SUPERFICIE = "#fcfcfb"

# Rampa sequencial continua (mesmo azul da rampa ordinal, degraus 100->700 da
# paleta) -- usada quando precisa de mais tons do que os 5 degraus nomeados
# comportam (ex: 50 barras).
_RAMPA_SEQUENCIAL = LinearSegmentedColormap.from_list("azul_seq", ["#cde2fb", "#0d366b"])


def _fonte_cjk():
    """cv2/matplotlib nao tem glifo CJK por padrao -- reaproveita fonte japonesa do projeto."""
    fontes = get_fonts_list()
    return fm.FontProperties(fname=fontes[0]) if fontes else None


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
    return {"csv": csv_path, "png": png_path, "faixas": faixas, "contagem": contagem}


def top_n1_kanji(contagem: Counter, out_dir: str = OUT_DIR, n: int = 50) -> dict:
    """
    Gráfico grande (barras horizontais, um kanji por linha) dos N kanji N1
    mais frequentes -- demonstração visual de quão rápido a frequência cai
    (cauda longa), pensado pra apresentação (não pra CSV, o `cobertura_n1`
    já cobre isso).
    """
    os.makedirs(out_dir, exist_ok=True)
    top = contagem.most_common(n)
    fonte_cjk = _fonte_cjk()

    fig_h = max(4.0, 0.32 * len(top))
    fig, ax = plt.subplots(figsize=(9, fig_h), facecolor=_SUPERFICIE)
    _estilo_eixo(ax)
    ax.xaxis.grid(True, color=_GRADE, linewidth=0.8, zorder=0)
    ax.yaxis.grid(False)

    ys = list(range(len(top)))
    valores = [v for _, v in top]
    cores = [_RAMPA_SEQUENCIAL(1 - i / max(1, len(top) - 1)) for i in range(len(top))]
    ax.barh(ys, valores, color=cores, height=0.72, zorder=3)
    ax.set_yticks(ys)
    ax.set_yticklabels([k for k, _ in top], fontproperties=fonte_cjk, fontsize=13)
    ax.invert_yaxis()  # mais frequente no topo
    ax.set_ylim(len(top) - 0.5, -0.5)

    for y, v in zip(ys, valores):
        ax.text(v + max(valores) * 0.012, y, str(v), va="center", ha="left",
                fontsize=8.5, color=_TINTA_SECUNDARIA)

    ax.set_xlabel("Ocorrências no corpus Manga109", fontsize=11, color=_TINTA_SECUNDARIA)
    ax.set_title(f"Top {len(top)} kanji N1 mais frequentes — queda acentuada de frequência",
                fontsize=15, color=_TINTA, fontweight="bold", pad=16, loc="left")
    fig.tight_layout()
    png_path = os.path.join(out_dir, f"top_{n}_kanji_n1.png")
    fig.savefig(png_path, dpi=150, facecolor=_SUPERFICIE)
    plt.close(fig)

    print(f"[INFO] {png_path}")
    return {"png": png_path}


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


def top_kanji_por_nivel(gt: dict, out_dir: str = OUT_DIR) -> dict:
    """
    O kanji MAIS frequente de cada nível JLPT (N1-N5), lado a lado -- um
    exemplo real e concreto por nível (não só estatística agregada), pra
    demonstrar a diferença de aparição de um jeito direto de ler.
    """
    os.makedirs(out_dir, exist_ok=True)
    niveis = ["n1", "n2", "n3", "n4", "n5"]
    fonte_cjk = _fonte_cjk()

    tops = []
    for nivel in niveis:
        kanjis = get_kanjis(nivel)
        contagem = contar_ocorrencias(set(kanjis), gt)
        kanji, ocorrencias = contagem.most_common(1)[0]
        tops.append({"nivel": nivel.upper(), "kanji": kanji, "ocorrencias": ocorrencias})

    fig, ax = plt.subplots(figsize=(7.5, 6), facecolor=_SUPERFICIE)
    _estilo_eixo(ax)
    nomes = [t["nivel"] for t in tops]
    valores = [t["ocorrencias"] for t in tops]
    bars = ax.bar(nomes, valores, color=_RAMPA_ORDINAL_N1_N5, width=0.6, zorder=3)
    ax.set_yscale("log")
    ax.set_ylim(1, max(valores) * 8)
    for bar, t in zip(bars, tops):
        cx = bar.get_x() + bar.get_width() / 2
        ax.text(cx, bar.get_height() * 1.15, f"{t['ocorrencias']}x",
                ha="center", va="bottom", fontsize=10.5, color=_TINTA_SECUNDARIA)
        ax.text(cx, bar.get_height() * 1.9, t["kanji"],
                ha="center", va="bottom", fontsize=30, color=_TINTA, fontproperties=fonte_cjk)
    ax.set_ylabel("Ocorrências no corpus (escala log)", fontsize=11, color=_TINTA_SECUNDARIA)
    ax.set_xlabel("Nível JLPT", fontsize=11, color=_TINTA_SECUNDARIA)
    ax.set_title("O kanji mais frequente de cada nível JLPT no Manga109", fontsize=14, color=_TINTA,
                fontweight="bold", pad=14, loc="left")
    fig.tight_layout()
    png_path = os.path.join(out_dir, "top_kanji_por_nivel.png")
    fig.savefig(png_path, dpi=150, facecolor=_SUPERFICIE)
    plt.close(fig)

    print(f"[INFO] {png_path}")
    return {"png": png_path, "tops": tops}


def comparar_niveis_pizza(linhas: list, out_dir: str = OUT_DIR) -> dict:
    """
    Mesmo dado de `comparar_niveis` (reaproveita `linhas`, não recalcula),
    em pizza -- fatia = participação de cada nível no total de ocorrências
    N1-N5 combinadas. Complementa o gráfico de barras (que mostra média por
    classe): aqui a pergunta é "de tudo que é N1-N5 no corpus, quanto é de
    cada nível", não "quão frequente é um kanji típico daquele nível".
    """
    os.makedirs(out_dir, exist_ok=True)
    nomes = [l["nivel"] for l in linhas]
    totais = [l["ocorrencias_totais"] for l in linhas]

    fig, ax = plt.subplots(figsize=(7, 7), facecolor=_SUPERFICIE)
    fig.patch.set_facecolor(_SUPERFICIE)
    wedges, _, autotexts = ax.pie(
        totais, colors=_CATEGORICA_5, startangle=90, counterclock=False,
        autopct=lambda pct: f"{pct:.0f}%", pctdistance=0.78,
        wedgeprops={"edgecolor": _SUPERFICIE, "linewidth": 2},
    )
    for t in autotexts:
        t.set_color("#ffffff")
        t.set_fontsize(11)
    ax.legend(
        wedges, [f"{n} ({t:,})".replace(",", ".") for n, t in zip(nomes, totais)],
        title="Nível JLPT (ocorrências)", loc="center left", bbox_to_anchor=(1.0, 0.5),
        frameon=False, fontsize=10.5, title_fontsize=10.5, labelcolor=_TINTA_SECUNDARIA,
    )
    ax.set_title("Participação de cada nível JLPT nas ocorrências N1-N5 no Manga109",
                fontsize=13.5, color=_TINTA, fontweight="bold", pad=16)
    n_classes_txt = " / ".join(f"{l['nivel']}={l['n_classes']}" for l in linhas)
    fig.text(0.5, 0.02,
            f"Fatia = soma bruta de ocorrências, não a média por kanji -- N1 tem muito mais classes\n"
            f"distintas que os outros níveis ({n_classes_txt} classes), então mesmo sendo o nível mais\n"
            f"raro POR kanji (ver gráfico de barras), a soma total ainda é grande.",
            ha="center", fontsize=9, color=_TINTA_MUTED)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    png_path = os.path.join(out_dir, "comparacao_niveis_pizza.png")
    fig.savefig(png_path, dpi=150, facecolor=_SUPERFICIE, bbox_inches="tight")
    plt.close(fig)

    print(f"[INFO] {png_path}")
    return {"png": png_path}


def main():
    gt = carregar_corpus()
    print(f"[INFO] Corpus carregado: {len(gt['paginas'])} páginas")
    res_n1 = cobertura_n1(gt)
    top_n1_kanji(res_n1["contagem"], n=50)
    top_kanji_por_nivel(gt)
    res_niveis = comparar_niveis(gt)
    comparar_niveis_pizza(res_niveis["linhas"])


if __name__ == "__main__":
    main()
