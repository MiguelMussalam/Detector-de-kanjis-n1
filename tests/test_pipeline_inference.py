"""
Smoke test do pipeline completo (detector + classificador) numa imagem real.
Pula automaticamente se os pesos não estiverem presentes no ambiente (não
são versionados por padrão em todo checkout, ex: CI sem os arquivos .pt).
"""

import cv2
import pytest

from src.pipeline.inference import Pipeline


def test_pipeline_predict_roda_sem_erro(pesos_disponiveis, imagem_benchmark):
    if not pesos_disponiveis:
        pytest.skip("Pesos do detector/classificador nao encontrados em weights/")

    pipeline = Pipeline()
    frame = cv2.imread(imagem_benchmark)
    assert frame is not None

    deteccoes = pipeline.predict(frame)

    assert isinstance(deteccoes, list)
    for det in deteccoes:
        assert len(det.bbox) == 4
        assert isinstance(det.kanji, str)
        assert 0.0 <= det.confianca_det <= 1.0
        assert 0.0 <= det.confianca_cls <= 1.0
