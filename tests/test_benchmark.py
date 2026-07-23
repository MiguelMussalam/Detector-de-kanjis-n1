"""
Smoke tests da lógica de match em src/helper/benchmark.py -- usa `Deteccao`
mockada (sem rodar detector/classificador de verdade) pra confirmar a
atribuição hit / miss-do-detector / miss-do-classificador.

`linha["transcricao"]` é o texto bruto da linha (mesmo campo salvo em
ground_truth.json/ground_truth_full.json) -- avaliar_pagina deriva o multiset
de kanji N1 esperados dele, filtrando pelo conjunto N1 real (_N1_SET). Por
isso os caracteres de teste precisam ser kanji N1 de verdade (não "愛"/"恋"/
"国"/"人" usados antes -- são básicos demais, N4/N5, e ficariam de fora do
filtro silenciosamente).
"""

from src.pipeline.inference import Deteccao
from src.helper.benchmark import avaliar_pagina

KANJI_A = chr(0x5200)  # kanji N1 real, so pra teste (nao importa o significado)
KANJI_B = chr(0x4E01)  # outro kanji N1 real, distinto de KANJI_A


def _det(bbox, kanji):
    return Deteccao(bbox=bbox, kanji=kanji, codepoint="", confianca_det=0.9, confianca_cls=0.9)


def test_hit_simples():
    linha = {"bbox": [0, 0, 100, 100], "transcricao": KANJI_A}
    deteccoes = [_det((10, 10, 20, 20), KANJI_A)]

    resultado = avaliar_pagina(deteccoes, [linha])

    assert resultado["hits"] == 1
    assert resultado["esperado"] == 1
    assert resultado["miss_detector"] == 0
    assert resultado["miss_classificador"] == 0


def test_miss_por_detector_quando_nao_ha_deteccao_na_regiao():
    linha = {"bbox": [0, 0, 100, 100], "transcricao": KANJI_A}
    deteccoes = []

    resultado = avaliar_pagina(deteccoes, [linha])

    assert resultado["hits"] == 0
    assert resultado["miss_detector"] == 1
    assert resultado["miss_classificador"] == 0


def test_miss_por_classificador_quando_ha_caixa_mas_classe_errada():
    linha = {"bbox": [0, 0, 100, 100], "transcricao": KANJI_A}
    deteccoes = [_det((10, 10, 20, 20), KANJI_B)]  # caixa existe na regiao, kanji previsto errado

    resultado = avaliar_pagina(deteccoes, [linha])

    assert resultado["hits"] == 0
    assert resultado["miss_detector"] == 0
    assert resultado["miss_classificador"] == 1


def test_deteccao_fora_da_regiao_da_linha_nao_conta():
    linha = {"bbox": [0, 0, 100, 100], "transcricao": KANJI_A}
    deteccoes = [_det((500, 500, 520, 520), KANJI_A)]  # centro fora da bbox da linha

    resultado = avaliar_pagina(deteccoes, [linha])

    assert resultado["hits"] == 0
    assert resultado["miss_detector"] == 1


def test_duas_esperadas_uma_acerta_outra_erra_com_candidata_sobrando():
    linha = {"bbox": [0, 0, 100, 100], "transcricao": KANJI_A + KANJI_B}
    deteccoes = [
        _det((10, 10, 20, 20), KANJI_A),   # acerta o primeiro esperado
        _det((30, 30, 40, 40), "OUTROS"),  # candidata sobrando pro segundo, classe errada
    ]

    resultado = avaliar_pagina(deteccoes, [linha])

    assert resultado["hits"] == 1
    assert resultado["miss_classificador"] == 1
    assert resultado["miss_detector"] == 0


def test_caractere_repetido_conta_cada_ocorrencia():
    """
    O bug real que motivou a correção: antes, `n1_esperados` era um set
    deduplicado -- uma linha com KANJI_A repetido 3x contava so 1 esperado
    (nao 3), e 1 unico acerto ja marcava 100%, mesmo com 2 ocorrencias reais
    faltando. Agora conta cada ocorrencia via multiset.
    """
    linha = {"bbox": [0, 0, 100, 100], "transcricao": KANJI_A * 3 + KANJI_B}
    deteccoes = [
        _det((10, 10, 20, 20), KANJI_A),
        _det((30, 30, 40, 40), KANJI_A),
        _det((50, 50, 60, 60), "OUTROS"),
        _det((70, 70, 80, 80), KANJI_B),
    ]

    resultado = avaliar_pagina(deteccoes, [linha])

    assert resultado["esperado"] == 4  # nao 2 (set deduplicado antigo)
    assert resultado["hits"] == 3
    assert resultado["miss_classificador"] == 1
    assert resultado["miss_detector"] == 0
