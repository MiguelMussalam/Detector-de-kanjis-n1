"""
Testes de src/detector/synth_page.py -- foco na subdivisao geometrica em
celulas (dividir_em_celulas), incluindo o suporte a grade 2D (colunas
multiplas dentro de uma mesma bbox <text>, comum em balao de fala do manga,
que antes da grade 2D era so descartado sem rotulo -- ver EXPERIMENTS.md/
plano da sessao pra contexto de por que essa mudanca foi feita).
"""

import random
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from config import DETSYN_CELL_MIN_PX, DETSYN_CELL_MAX_PX
from src.detector.synth_page import dividir_em_celulas, processar_pagina_para_synth


def test_coluna_unica_identico_ao_comportamento_atual():
    """
    Caso legado (linha curta, coluna unica) -- grupos deve sair 1 e o
    resultado tem que ser IGUAL (nao so aproximado) ao calculo manual de
    antes da grade 2D existir, pra garantir zero regressao no caso que ja
    funcionava.
    """
    bbox = (0, 0, 25, 180)
    n_chars = 9
    resultado = dividir_em_celulas(bbox, n_chars)

    assert resultado is not None
    assert len(resultado) == n_chars
    lado_celula = 180 / 9
    esperado = [(0, i * lado_celula, 25, (i + 1) * lado_celula) for i in range(9)]
    assert resultado == esperado


def test_multi_coluna_agora_e_aceita():
    """
    Balao de fala com 3 colunas de 12 caracteres dentro da mesma bbox --
    antes da grade 2D isso caia em "celula_pequena_demais" (180/36=6px<8px)
    e a linha inteira era descartada sem rotulo.
    """
    bbox = (0, 0, 65, 216)
    n_chars = 36
    resultado = dividir_em_celulas(bbox, n_chars)

    assert resultado is not None
    assert len(resultado) == n_chars
    for x1, y1, x2, y2 in resultado:
        assert DETSYN_CELL_MIN_PX <= (x2 - x1) <= DETSYN_CELL_MAX_PX
        assert DETSYN_CELL_MIN_PX <= (y2 - y1) <= DETSYN_CELL_MAX_PX


def test_geometria_degenerada_ainda_e_rejeitada():
    """200 caracteres numa bbox de 40x40 -- nem o melhor grid cabe no piso de 8px."""
    resultado = dividir_em_celulas((0, 0, 40, 40), 200)
    assert resultado is None


@pytest.mark.parametrize("bbox,n_chars", [
    ((0, 0, 25, 180), 9),     # legado, coluna unica
    ((0, 0, 65, 216), 36),    # resgatado, multi-coluna
    ((0, 0, 90, 100), 10),    # resgatado, multi-coluna (grupos=3, resto)
])
def test_grade_sempre_produz_exatamente_n_chars_celulas(bbox, n_chars):
    resultado = dividir_em_celulas(bbox, n_chars)
    assert resultado is not None
    assert len(resultado) == n_chars


def test_ultima_coluna_pode_ter_menos_que_por_grupo():
    """
    bbox 90x100, 10 caracteres -> grupos=3, por_grupo=4 (3+3+4 nao bate,
    verificado: grupos_raw=sqrt(10*90/100)=3.0 -> grupos=3, por_grupo=ceil(10/3)=4,
    recorrigido grupos=ceil(10/4)=3). A ultima coluna (primeira lida, mais a
    direita fica com 4; a mais a esquerda, ultima na leitura, fica com resto
    10-2*4=2) tem menos que por_grupo=4 caracteres.
    """
    bbox = (0, 0, 90, 100)
    n_chars = 10
    resultado = dividir_em_celulas(bbox, n_chars)

    assert resultado is not None
    assert len(resultado) == n_chars

    faixas_x = sorted(set((round(x1, 4), round(x2, 4)) for x1, y1, x2, y2 in resultado))
    assert len(faixas_x) == 3  # 3 colunas distintas

    contagem_por_faixa = {}
    for x1, y1, x2, y2 in resultado:
        chave = (round(x1, 4), round(x2, 4))
        contagem_por_faixa[chave] = contagem_por_faixa.get(chave, 0) + 1
    assert sorted(contagem_por_faixa.values()) == [2, 4, 4]


def test_celulas_geram_cell_w_cell_h_inteiros_positivos():
    """Contrato que renderizar_glifo_para_celula/_renderizar_tentativa esperam."""
    resultado = dividir_em_celulas((0, 0, 65, 216), 36)
    assert resultado is not None
    for x1, y1, x2, y2 in resultado:
        cell_w = max(1, int(round(x2 - x1)))
        cell_h = max(1, int(round(y2 - y1)))
        assert cell_w >= 1
        assert cell_h >= 1


def test_processar_pagina_para_synth_integra_grade_multicoluna(fonte_disponivel, kanji_n1_exemplo):
    """
    Integracao: uma bbox multi-coluna de verdade, processada pela pipeline
    inteira. Antes da grade 2D isso resultava em bboxes_saida VAZIO (linha
    inteira pulada, so tinta apagada) -- e' o teste de maior valor porque
    exercita o contrato completo que generate_pages.py depende.
    """
    frame_bgr = np.full((300, 300, 3), 255, dtype=np.uint8)
    n_chars = 36
    text_el = ET.Element("text", {"xmin": "0", "ymin": "0", "xmax": "65", "ymax": "216"})
    text_el.text = kanji_n1_exemplo * n_chars

    rng = random.Random(42)
    frame_resultado, bboxes_saida = processar_pagina_para_synth(
        frame_bgr, [text_el], [kanji_n1_exemplo], [fonte_disponivel], rng
    )

    assert frame_resultado.shape == frame_bgr.shape
    assert len(bboxes_saida) >= n_chars // 2  # antes da grade 2D, isso era 0
