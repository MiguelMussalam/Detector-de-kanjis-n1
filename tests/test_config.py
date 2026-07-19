"""
Testes de config.py -- foco em _buscar_manga109_diretorio, que teve um bug
real nesta sessao: a busca retornava no primeiro match "fraco" (pasta com
"manga109" no nome, tipico de datasets Kaggle nao relacionados como o
Roboflow "manga109-character-bouding-box") sem terminar de varrer a arvore,
entao um match "forte" de verdade (pasta com images/ E annotations/ juntas)
que existia em outro lugar nunca era encontrado se a varredura passasse
primeiro pelo match fraco. Isso gerou 0 paginas sinteticas silenciosamente
num treino real.
"""

import os

from config import _buscar_manga109_diretorio


def _sem_manga109_local(monkeypatch):
    """Faz os.path.exists mentir pra pasta local, forcando a busca cair no Kaggle."""
    monkeypatch.setattr(os.path, "exists", lambda p: p == "/kaggle/input")


def test_prefere_match_forte_mesmo_quando_match_fraco_aparece_primeiro(monkeypatch):
    """Bug real: Roboflow (match fraco) vinha antes do corpus completo (match forte) na varredura."""
    _sem_manga109_local(monkeypatch)

    def fake_walk(root):
        yield ("/kaggle/input/datasets/miguelmussalam/manga109-character-bouding-box/test", ["images"], [])
        yield ("/kaggle/input/datasets/miguelmussalam/manga109/Manga109", ["images", "annotations"], [])

    monkeypatch.setattr(os, "walk", fake_walk)

    resultado = _buscar_manga109_diretorio()
    assert resultado == "/kaggle/input/datasets/miguelmussalam/manga109/Manga109"


def test_prefere_match_forte_quando_ele_aparece_primeiro(monkeypatch):
    """Ordem inversa -- garante que o match forte tambem vence quando vem primeiro."""
    _sem_manga109_local(monkeypatch)

    def fake_walk(root):
        yield ("/kaggle/input/datasets/miguelmussalam/manga109/Manga109", ["images", "annotations"], [])
        yield ("/kaggle/input/datasets/miguelmussalam/manga109-character-bouding-box/test", ["images"], [])

    monkeypatch.setattr(os, "walk", fake_walk)

    resultado = _buscar_manga109_diretorio()
    assert resultado == "/kaggle/input/datasets/miguelmussalam/manga109/Manga109"


def test_cai_no_match_fraco_quando_nao_ha_match_forte_em_lugar_nenhum(monkeypatch):
    """Sem nenhum match forte na arvore inteira, o fallback fraco ainda precisa funcionar."""
    _sem_manga109_local(monkeypatch)

    def fake_walk(root):
        yield ("/kaggle/input/datasets/miguelmussalam/manga109-character-bouding-box/test", ["images"], [])

    monkeypatch.setattr(os, "walk", fake_walk)

    resultado = _buscar_manga109_diretorio()
    assert resultado == "/kaggle/input/datasets/miguelmussalam/manga109-character-bouding-box/test"
