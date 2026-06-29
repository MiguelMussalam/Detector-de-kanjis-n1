import random
import numpy as np
import cv2
import albumentations as A
from PIL import Image

from config import (
    # Morfologia
    EROSAO_PROB, DILATACAO_PROB, MORFO_KERNEL, MORFO_ITERACOES,
    # Ruído de scanner
    GAUSS_NOISE_PROB, GAUSS_NOISE_STD_MIN, GAUSS_NOISE_STD_MAX,
    SALT_PEPPER_PROB, SALT_PEPPER_AMOUNT,
    BLUR_PROB, BLUR_KERNEL,
    # Distorções geométricas
    ELASTIC_PROB, ELASTIC_ALPHA, ELASTIC_SIGMA,
    GRID_PROB, GRID_DISTORT_LIMIT,
    ROTATE_PROB, ROTATE_LIMIT,
)


_pipeline_geometrica = A.Compose([
    A.ElasticTransform(
        alpha=ELASTIC_ALPHA,
        sigma=ELASTIC_SIGMA,
        interpolation=cv2.INTER_LINEAR,
        border_mode=cv2.BORDER_REPLICATE,
        p=ELASTIC_PROB,
    ),
    A.GridDistortion(
        distort_limit=(-GRID_DISTORT_LIMIT, GRID_DISTORT_LIMIT),
        interpolation=cv2.INTER_LINEAR,
        border_mode=cv2.BORDER_REPLICATE,
        p=GRID_PROB,
    ),
    A.ShiftScaleRotate(
        shift_limit=0.0,
        scale_limit=0.0,
        rotate_limit=(-ROTATE_LIMIT, ROTATE_LIMIT),
        interpolation=cv2.INTER_LINEAR,
        border_mode=cv2.BORDER_REPLICATE,
        p=ROTATE_PROB,
    ),
])

_pipeline_scanner = A.Compose([
    A.GaussNoise(
        std_range=(GAUSS_NOISE_STD_MIN, GAUSS_NOISE_STD_MAX),
        mean_range=(0.0, 0.0),
        per_channel=True,
        p=GAUSS_NOISE_PROB,
    ),
    A.SaltAndPepper(
        amount=SALT_PEPPER_AMOUNT,
        p=SALT_PEPPER_PROB,
    ),
])


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------

def _pil_to_np(imagem: Image.Image) -> np.ndarray:
    """Converte PIL RGB -> numpy uint8 HxWx3."""
    return np.array(imagem.convert("RGB"), dtype=np.uint8)


def _np_to_pil(arr: np.ndarray) -> Image.Image:
    """Converte numpy uint8 HxWx3 -> PIL RGB."""
    return Image.fromarray(arr.astype(np.uint8), mode="RGB")


def _aplicar_morfologia(arr: np.ndarray) -> np.ndarray:
    """
    Aplica erosao XOR dilatacao (nunca ambas no mesmo sample).
    Erosao    -> tracados mais finos / falta de tinta (papel poroso antigo).
    Dilatacao -> tracados mais grossos / sangramento de tinta (ink bleed).
    """
    kernel = np.ones((MORFO_KERNEL, MORFO_KERNEL), dtype=np.uint8)
    r = random.random()

    if r < EROSAO_PROB:
        return cv2.erode(arr, kernel, iterations=MORFO_ITERACOES)

    if r < EROSAO_PROB + DILATACAO_PROB:
        return cv2.dilate(arr, kernel, iterations=MORFO_ITERACOES)

    return arr  # sem morfologia neste sample


def _aplicar_blur(arr: np.ndarray) -> np.ndarray:
    """Desfoque gaussiano leve simulando perda de foco do scanner."""
    if random.random() < BLUR_PROB:
        k = BLUR_KERNEL if BLUR_KERNEL % 2 == 1 else BLUR_KERNEL + 1
        return cv2.GaussianBlur(arr, (k, k), sigmaX=0)
    return arr


# ---------------------------------------------------------------------------
# API publica
# ---------------------------------------------------------------------------

def aplicar_degradacoes(imagem: Image.Image) -> Image.Image:
    """
    Aplica o pipeline completo de degradacao 2D a uma imagem PIL.

    Ordem cronologica fisica:
      1. Distorcoes geometricas  (curvatura/desalinhamento da pagina ao escanear)
      2. Morfologia de tinta     (erosao XOR dilatacao -- defeitos de impressao)
      3. Ruido de scanner        (GaussNoise + SaltAndPepper + Blur -- sensor optico)

    Retorna uma nova imagem PIL com as mesmas dimensoes da entrada.
    """
    arr = _pil_to_np(imagem)

    # 1 — Geometria
    arr = _pipeline_geometrica(image=arr)["image"]

    # 2 — Morfologia (tinta)
    arr = _aplicar_morfologia(arr)

    # 3 — Scanner: ruido estocastico, depois blur (ordem importa)
    arr = _pipeline_scanner(image=arr)["image"]
    arr = _aplicar_blur(arr)

    return _np_to_pil(arr)
