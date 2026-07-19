"""
Smoke tests do gerador sintético do classificador (src/classifier/generate_crops.py).
Cobre o que já mordeu a gente nesta sessão sem teste automatizado: canvas
saturado (glifo "tofu"/mancha sólida) tem que ser pego pelo fill_ratio, e
`generate_sample` sempre tem que devolver algo com o tamanho certo mesmo
quando precisa cair no fallback.
"""

import random

import numpy as np

from config import CLASSIFIER_INPUT_SIZE, CLASSIFIER_CANVAS_MARGIN
from src.classifier.generate_crops import render_glyph, generate_sample, _fill_ratio_glifo


def test_render_glyph_produz_imagem_com_tinta(fonte_disponivel, kanji_n1_exemplo):
    canvas_size = 64
    img = render_glyph(kanji_n1_exemplo, fonte_disponivel, font_size=32, canvas_size=canvas_size)

    assert img.shape == (canvas_size, canvas_size)
    assert img.dtype == np.uint8
    assert img.min() < 255, "glifo nao desenhou nenhuma tinta -- canvas ficou em branco"


def test_fill_ratio_canvas_totalmente_branco_e_zero():
    """Mascara vazia (nada abaixo do limiar) -- caso da "regiao inteira e fundo"."""
    canvas = np.full((64, 64), 255, dtype=np.uint8)
    mask = canvas <= 240
    assert _fill_ratio_glifo(canvas, mask) == 0.0


def test_fill_ratio_canvas_totalmente_preto_e_zero():
    """
    Mascara cobrindo a imagem inteira (nada de fundo pra comparar) tambem
    retorna 0.0 -- mesmo resultado do caso vazio, por um caminho diferente
    (_fg_bg_means nao tem o que comparar). Isso e o que faz um "tofu box"
    (fonte sem glifo, canvas satura solido) cair fora da faixa aceita por
    _legivel() (CLF_MIN_FILL_RATIO=0.12) e disparar o retry -- confirmado
    empiricamente antes de escrever este teste, nao assumido.
    """
    canvas = np.full((64, 64), 0, dtype=np.uint8)
    mask = canvas <= 240
    assert _fill_ratio_glifo(canvas, mask) == 0.0


def test_generate_sample_produz_imagem_do_tamanho_certo(fonte_disponivel, kanji_n1_exemplo):
    rng = random.Random(42)
    img = generate_sample(
        char=kanji_n1_exemplo, font_path=fonte_disponivel, font_size=28, rng=rng,
        output_size=CLASSIFIER_INPUT_SIZE, canvas_margin=CLASSIFIER_CANVAS_MARGIN,
        fonts_fallback=[fonte_disponivel],
    )
    assert img.shape == (CLASSIFIER_INPUT_SIZE, CLASSIFIER_INPUT_SIZE)
    assert img.dtype == np.uint8
