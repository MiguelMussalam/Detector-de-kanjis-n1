"""
Smoke tests da lógica de match em src/helper/benchmark.py -- usa `Deteccao`
mockada (sem rodar detector/classificador de verdade) pra confirmar a
atribuição hit / miss-do-detector / miss-do-classificador.
"""

from src.pipeline.inference import Deteccao
from src.helper.benchmark import avaliar_pagina


def _det(bbox, kanji):
    return Deteccao(bbox=bbox, kanji=kanji, codepoint="", confianca_det=0.9, confianca_cls=0.9)


def test_hit_simples():
    linha = {"bbox": [0, 0, 100, 100], "n1_esperados": ["愛"]}
    deteccoes = [_det((10, 10, 20, 20), "愛")]

    resultado = avaliar_pagina(deteccoes, [linha])

    assert resultado["hits"] == 1
    assert resultado["esperado"] == 1
    assert resultado["miss_detector"] == 0
    assert resultado["miss_classificador"] == 0


def test_miss_por_detector_quando_nao_ha_deteccao_na_regiao():
    linha = {"bbox": [0, 0, 100, 100], "n1_esperados": ["愛"]}
    deteccoes = []

    resultado = avaliar_pagina(deteccoes, [linha])

    assert resultado["hits"] == 0
    assert resultado["miss_detector"] == 1
    assert resultado["miss_classificador"] == 0


def test_miss_por_classificador_quando_ha_caixa_mas_classe_errada():
    linha = {"bbox": [0, 0, 100, 100], "n1_esperados": ["愛"]}
    deteccoes = [_det((10, 10, 20, 20), "恋")]  # caixa existe na regiao, kanji previsto errado

    resultado = avaliar_pagina(deteccoes, [linha])

    assert resultado["hits"] == 0
    assert resultado["miss_detector"] == 0
    assert resultado["miss_classificador"] == 1


def test_deteccao_fora_da_regiao_da_linha_nao_conta():
    linha = {"bbox": [0, 0, 100, 100], "n1_esperados": ["愛"]}
    deteccoes = [_det((500, 500, 520, 520), "愛")]  # centro fora da bbox da linha

    resultado = avaliar_pagina(deteccoes, [linha])

    assert resultado["hits"] == 0
    assert resultado["miss_detector"] == 1


def test_duas_esperadas_uma_acerta_outra_erra_com_candidata_sobrando():
    linha = {"bbox": [0, 0, 100, 100], "n1_esperados": ["愛", "恋"]}
    deteccoes = [
        _det((10, 10, 20, 20), "愛"),   # acerta o primeiro esperado
        _det((30, 30, 40, 40), "OUTROS"),  # candidata sobrando pro segundo, classe errada
    ]

    resultado = avaliar_pagina(deteccoes, [linha])

    assert resultado["hits"] == 1
    assert resultado["miss_classificador"] == 1
    assert resultado["miss_detector"] == 0
