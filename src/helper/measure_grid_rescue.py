"""
measure_grid_rescue.py
========================
Mede, no corpus Manga109 inteiro, quantas linhas <text> hoje descartadas por
dividir_em_celulas() (geometria de coluna única, ver src/detector/synth_page.py)
seriam resgatadas por uma extensão de grade 2D (múltiplas colunas) -- ANTES de
implementar a mudança de verdade, pra confirmar que a matemática funciona
contra dado real e não produz grade degenerada.

Reproduz a lógica ATUAL (com motivo de rejeição) pra confirmar a baseline já
medida nesta investigação (24.8% ok / 30.2% celula pequena / 1.8% celula
grande / 40.3% razao de largura / 3.0% vazia) e a lógica NOVA (estimativa de
grade C colunas x R linhas) lado a lado, sobre o mesmo conjunto de linhas.

Uso:
    python -m src.helper.measure_grid_rescue [--limit-volumes N]
"""

import argparse
import math
import os
from collections import Counter
from glob import glob

import numpy as np
from tqdm import tqdm

from config import (
    MANGA109_ANNOTATIONS, MANGA109_ALIGN_VAL_VOLUMES,
    DETSYN_CELL_MIN_PX, DETSYN_CELL_MAX_PX, DETSYN_MAX_RAZAO_LARGURA,
)
from src.helper.manga109_corpus import iter_paginas
from src.helper.manga109_align import limpar_string


def _listar_volumes(limit: int = None) -> list:
    todos = sorted(
        os.path.splitext(os.path.basename(f))[0]
        for f in glob(os.path.join(MANGA109_ANNOTATIONS, "*.xml"))
    )
    volumes = [v for v in todos if v not in MANGA109_ALIGN_VAL_VOLUMES]
    return volumes[:limit] if limit else volumes


def categorizar_atual(n_chars: int, eixo_leitura: float, eixo_perp: float) -> str:
    """Reproduz dividir_em_celulas() de hoje (coluna/linha unica), so com motivo."""
    lado_celula = eixo_leitura / n_chars
    if lado_celula < DETSYN_CELL_MIN_PX:
        return "celula_pequena_demais"
    if lado_celula > DETSYN_CELL_MAX_PX:
        return "celula_grande_demais"
    if eixo_perp > lado_celula * DETSYN_MAX_RAZAO_LARGURA:
        return "razao_largura_multicoluna"
    return "ok"


def estimar_grade(n_chars: int, eixo_leitura: float, eixo_perp: float):
    """
    Estima (grupos, por_grupo) assumindo celulas ~quadradas: grupos = numero
    de colunas (vertical) ou linhas (horizontal) lado a lado; por_grupo =
    caracteres por grupo. Replica a proposta de plano pra synth_page.py
    (ainda nao implementada la) -- ver plano aprovado.
    """
    if n_chars <= 0 or eixo_leitura <= 0 or eixo_perp <= 0:
        return None
    grupos_raw = math.sqrt(n_chars * eixo_perp / eixo_leitura)
    grupos = min(max(1, int(grupos_raw + 0.5)), n_chars)  # arredonda p/ cima em .5
    por_grupo = math.ceil(n_chars / grupos)
    grupos = math.ceil(n_chars / por_grupo)  # corrige grupo vazio no final
    return grupos, por_grupo


def categorizar_novo(n_chars: int, eixo_leitura: float, eixo_perp: float):
    """Retorna (motivo, grupos, lado_celula, cell_perp) pra logica de grade 2D."""
    grade = estimar_grade(n_chars, eixo_leitura, eixo_perp)
    if grade is None:
        return "rejeitado_geometria_invalida", None, None, None
    grupos, por_grupo = grade
    lado_celula = eixo_leitura / por_grupo
    cell_perp = eixo_perp / grupos

    if not (DETSYN_CELL_MIN_PX <= lado_celula <= DETSYN_CELL_MAX_PX):
        return "rejeitado_lado_celula", grupos, lado_celula, cell_perp
    if grupos == 1:
        if cell_perp > lado_celula * DETSYN_MAX_RAZAO_LARGURA:
            return "rejeitado_razao", grupos, lado_celula, cell_perp
    else:
        if not (DETSYN_CELL_MIN_PX <= cell_perp <= DETSYN_CELL_MAX_PX):
            return "rejeitado_cell_perp", grupos, lado_celula, cell_perp
    return "ok", grupos, lado_celula, cell_perp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit-volumes", type=int, default=None)
    args = parser.parse_args()

    volumes = _listar_volumes(args.limit_volumes)

    cat_atual = Counter()
    cat_novo = Counter()
    cruzamento = Counter()   # (motivo_atual, passou_novo) -> contagem
    grupos_hist = Counter()
    lado_celula_novos = []
    cell_perp_novos = []
    orientacao_resgatadas = Counter()
    regressao_falhas = 0
    n_linhas = 0

    for volume, idx, page in tqdm(iter_paginas(volumes), desc="paginas"):
        for text_el in page.iter("text"):
            texto_limpo = limpar_string(text_el.text)
            if not texto_limpo:
                cat_atual["vazia"] += 1
                cat_novo["vazia"] += 1
                continue

            x1, y1 = int(text_el.get("xmin")), int(text_el.get("ymin"))
            x2, y2 = int(text_el.get("xmax")), int(text_el.get("ymax"))
            largura, altura = x2 - x1, y2 - y1
            if largura <= 0 or altura <= 0:
                cat_atual["bbox_invalida"] += 1
                cat_novo["bbox_invalida"] += 1
                continue

            n_chars = len(texto_limpo)
            vertical = altura >= largura
            eixo_leitura = altura if vertical else largura
            eixo_perp = largura if vertical else altura

            n_linhas += 1
            motivo_atual = categorizar_atual(n_chars, eixo_leitura, eixo_perp)
            cat_atual[motivo_atual] += 1

            motivo_novo, grupos, lado_celula, cell_perp = categorizar_novo(n_chars, eixo_leitura, eixo_perp)
            cat_novo[motivo_novo] += 1
            cruzamento[(motivo_atual, motivo_novo == "ok")] += 1

            if motivo_novo == "ok":
                grupos_hist[grupos] += 1
                lado_celula_novos.append(lado_celula)
                cell_perp_novos.append(cell_perp)
                if motivo_atual != "ok":
                    orientacao_resgatadas["vertical" if vertical else "horizontal"] += 1

            # Regressao: toda linha "ok" hoje precisa continuar "ok" com grupos==1
            # e celulas identicas (nao so aproximadas) -- prova empirica de que a
            # nova logica nao muda nada pro caso que ja funciona.
            if motivo_atual == "ok":
                lado_atual = eixo_leitura / n_chars
                identico = (
                    motivo_novo == "ok" and grupos == 1
                    and math.isclose(lado_celula, lado_atual) and math.isclose(cell_perp, eixo_perp)
                )
                if not identico:
                    regressao_falhas += 1

    if n_linhas == 0:
        print("[AVISO] Nenhuma linha avaliada.")
        return

    print(f"\n[INFO] {n_linhas} linhas <text> avaliadas (volumes de validacao excluidos)")

    print("\n=== Categorizacao ATUAL (baseline) ===")
    for k, v in cat_atual.most_common():
        print(f"  {k:<30} {v:>7} ({100 * v / n_linhas:.1f}%)")

    print("\n=== Categorizacao NOVA (grade 2D proposta) ===")
    for k, v in cat_novo.most_common():
        print(f"  {k:<30} {v:>7} ({100 * v / n_linhas:.1f}%)")

    print("\n=== Taxa de resgate por categoria antiga ===")
    for motivo in ["celula_pequena_demais", "celula_grande_demais", "razao_largura_multicoluna"]:
        total = cruzamento[(motivo, True)] + cruzamento[(motivo, False)]
        resgatadas = cruzamento[(motivo, True)]
        if total:
            print(f"  {motivo:<30} {resgatadas}/{total} resgatadas ({100 * resgatadas / total:.1f}%)")

    print(f"\n[INFO] Falhas de regressao (linha 'ok' hoje que mudou com grupos==1): {regressao_falhas}")

    print("\n=== Histograma de 'grupos' (entre as que passam na logica nova) ===")
    for g, c in sorted(grupos_hist.items()):
        print(f"  grupos={g:<4} {c:>7}")

    if lado_celula_novos:
        arr_lado = np.array(lado_celula_novos)
        arr_perp = np.array(cell_perp_novos)
        percentis = [5, 25, 50, 75, 95]
        print("\n=== Distribuicao de tamanho de celula (px, entre as que passam) ===")
        print("lado_celula p5/p25/p50/p75/p95:", [round(float(np.percentile(arr_lado, p)), 1) for p in percentis])
        print("cell_perp   p5/p25/p50/p75/p95:", [round(float(np.percentile(arr_perp, p)), 1) for p in percentis])

    total_resgatadas = sum(orientacao_resgatadas.values())
    print("\n=== Orientacao das linhas recem-resgatadas (antes descartadas, agora ok) ===")
    for orient, c in orientacao_resgatadas.most_common():
        pct = f" ({100 * c / total_resgatadas:.1f}%)" if total_resgatadas else ""
        print(f"  {orient:<12} {c:>7}{pct}")


if __name__ == "__main__":
    main()
