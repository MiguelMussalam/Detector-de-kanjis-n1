"""
Smoke tests do filtro heurístico de transcrição corrompida
(src/helper/manga109_corpus.py). O caso de script estrangeiro isolado é
literalmente o bug real desta sessão: a primeira versão usava whitelist de
pontuação e rejeitava 57/58 linhas legítimas (aspas curvas, travessão,
coração...) como "corrompidas" -- a versão corrigida (blacklist de blocos
Unicode) precisa continuar aceitando essa tipografia normal.
"""

from config import CORPUS_CELL_AREA_MIN
from src.helper.manga109_corpus import avaliar_transcricao_confiavel


def test_transcricao_vazia_e_reprovada():
    confiavel, motivo = avaliar_transcricao_confiavel("", [0, 0, 100, 100])
    assert not confiavel
    assert motivo == "transcricao_vazia"


def test_script_estrangeiro_isolado_e_reprovado():
    """Caso real achado nesta sessão: letra grega isolada em efeito sonoro corrompido."""
    texto = "天秤座β星ーズベンエルシャマリ"
    confiavel, motivo = avaliar_transcricao_confiavel(texto, [0, 0, 400, 400])
    assert not confiavel
    assert motivo == "script_anomalo"


def test_pontuacao_tipografica_normal_nao_e_rejeitada():
    """
    Aspas curvas, travessão, coração -- tipografia normal de mangá, não
    corrupção. Bug real corrigido nesta sessão (whitelist rejeitava isso).
    """
    texto = "「愛してる」〜って言ったの♡"
    bbox = [0, 0, 200, 400]  # area generosa por caractere, nao deve reprovar por tamanho
    confiavel, motivo = avaliar_transcricao_confiavel(texto, bbox)
    assert confiavel
    assert motivo is None


def test_bbox_pequena_demais_e_reprovada():
    texto = "五文字exato"  # varios caracteres
    bbox = [0, 0, 10, 10]  # area=100, bem abaixo do piso por caractere
    confiavel, motivo = avaliar_transcricao_confiavel(texto, bbox)
    assert not confiavel
    assert motivo == "bbox_pequena_demais"


def test_bbox_generosa_passa_no_piso_de_area():
    texto = "愛"
    lado = (CORPUS_CELL_AREA_MIN * 4) ** 0.5 + 5  # area = 4x o piso, com folga
    bbox = [0, 0, lado, lado]
    confiavel, motivo = avaliar_transcricao_confiavel(texto, bbox)
    assert confiavel
    assert motivo is None
