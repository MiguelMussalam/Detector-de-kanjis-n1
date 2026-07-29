"""
grade_visual.py
================
Primitiva de plotagem compartilhada: salva uma grade de exemplos (imagem +
título colorido) em PNG. Extraída de duas implementações que tinham o mesmo
código matplotlib duplicado (`eval_confusoes` em `src/classifier/eval.py` e
a antiga `salvar_grade_auditoria` em `src/helper/ocr_baseline_compare.py`).

Cada chamador continua montando a lista de itens do seu jeito (imagem em
disco vs. recorte em memória, título fixo vs. condicional, cor fixa vs.
condicional) -- essa função só cuida da parte genérica: amostragem, fonte
CJK, layout do grid, salvar em arquivo.

Uso:
    from src.helper.grade_visual import salvar_grade_visual
    salvar_grade_visual(itens, "data/comparacao_visual/nosso_pipeline.png")
"""

import os
import random


def salvar_grade_visual(itens: list, out_path: str, n: int = 24, seed: int = 42,
                        cols: int = 4, figsize_cel: tuple = (4.5, 3.2), fontsize: int = 8):
    """
    Salva uma grade de `itens` amostrados em `out_path`.

    Cada item é um dict: {"imagem": PIL.Image ou ndarray, "titulo": str,
    "cor": str, "cmap": str|None (opcional)}.
    """
    if not itens:
        print("[AVISO] Nenhuma amostra pra grade visual.")
        return None

    import matplotlib.font_manager as fm
    import matplotlib.pyplot as plt
    from src.helper.fonts import get_fonts_list

    # Fonte padrao do matplotlib (DejaVu Sans) nao tem glifo CJK -- titulo com
    # kanji sairia com caixa vazia. Reaproveita uma das fontes japonesas ja
    # baixadas em assets/fonts/.
    fontes = get_fonts_list()
    fonte_cjk = fm.FontProperties(fname=fontes[0]) if fontes else None

    rng = random.Random(seed)
    amostra = rng.sample(itens, min(n, len(itens)))
    rows = (len(amostra) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(figsize_cel[0] * cols, figsize_cel[1] * rows))
    axes_flat = list(axes.flat) if len(amostra) > 1 else [axes]

    for ax, item in zip(axes_flat, amostra):
        ax.imshow(item["imagem"], cmap=item.get("cmap"))
        ax.set_title(item["titulo"], color=item["cor"], fontsize=fontsize, fontproperties=fonte_cjk)
        ax.axis("off")

    for ax in axes_flat[len(amostra):]:
        ax.axis("off")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[INFO] Grade visual salva em: {out_path}")
    return out_path
