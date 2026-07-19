"""
Fixtures compartilhadas pelos testes. Pula (nao falha) testes que dependem
de recursos opcionais do ambiente (fontes baixadas, pesos treinados) --
esses recursos sao gerados/baixados fora do git (assets/fonts/, weights/),
entao um checkout novo do repo nao os tem por padrao.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import ROOT_DIR, DETECTOR_WEIGHTS_PATH, CLF_WEIGHTS_PATH
from src.helper.fonts import get_fonts_list
from src.helper.kanjis import get_kanjis


@pytest.fixture(scope="session")
def fonte_disponivel():
    fontes = get_fonts_list()
    if not fontes:
        pytest.skip("Nenhuma fonte em assets/fonts/ -- rode `python -m src.helper.fonts` primeiro")
    return fontes[0]


@pytest.fixture(scope="session")
def kanji_n1_exemplo():
    kanjis = get_kanjis("n1")
    if not kanjis:
        pytest.skip("Lista de kanji N1 vazia/indisponivel")
    return kanjis[0]


@pytest.fixture(scope="session")
def pesos_disponiveis():
    return os.path.exists(DETECTOR_WEIGHTS_PATH) and os.path.exists(CLF_WEIGHTS_PATH)


@pytest.fixture(scope="session")
def imagem_benchmark():
    caminho = os.path.join(ROOT_DIR, "data", "benchmark", "pagina_UchuKigekiM774_59.jpg")
    if not os.path.exists(caminho):
        pytest.skip(f"Imagem de benchmark nao encontrada: {caminho}")
    return caminho
