"""
harvest_backgrounds.py
======================
Coleta patches de fundo REAL de páginas do Manga109 (screentone, hachura,
traço de arte — sem nenhum glifo por perto) para usar como fundo no gerador
sintético (`src/classifier/generate_crops.py`), no lugar do "papel liso"
simulado (valor sólido + ruído gaussiano leve).

Cada patch é maior que o input do classificador (`CLF_BG_PATCH_SIZE`, ex:
96 > 64) de propósito: o gerador sintético recorta um sub-patch aleatório
de cada arquivo a cada uso, então mesmo um pool de poucos milhares de
arquivos gera muito mais variedade efetiva do que se cada arquivo fosse
usado sempre do mesmo jeito.

Sem rótulo nenhum envolvido — só estatística/textura de fundo.

Uso:
    python -m src.helper.harvest_backgrounds
"""

import os
import random

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from config import (
    BACKGROUNDS_REAL_DIR, CLF_BG_PATCH_SIZE, CLF_BG_HARVEST_COUNT, CLF_BG_HARVEST_PAGINAS,
    CLF_BG_MAX_DIST_FACTOR, CLF_BG_MAX_STD,
    CLF_BG_DARK_MEAN_THRESHOLD, CLF_BG_HARVEST_COUNT_ESCURO,
    DETECTOR_WEIGHTS_PATH, PIPELINE_DET_CONF, PIPELINE_DET_IOU, PIPELINE_DET_MAX_DET,
)
from src.helper.measure_real_stats import listar_paginas, _sobrepoe
from src.pipeline.inference import load_detector


def _perto_de_algum_box(cx, cy, boxes, distancia_max) -> bool:
    """
    True se (cx, cy) esta a uma distancia razoavel de pelo menos um glifo
    detectado -- usado pra puxar a amostragem de fundo pra perto de onde tem
    texto de verdade (tipicamente dentro de balao, fundo mais liso), em vez
    de qualquer canto da pagina (arte do painel, que costuma ser bem mais
    carregada visualmente do que o fundo atras de um caractere).
    """
    if not boxes:
        return False
    for (x1, y1, x2, y2) in boxes:
        bcx, bcy = (x1 + x2) / 2, (y1 + y2) / 2
        if ((cx - bcx) ** 2 + (cy - bcy) ** 2) ** 0.5 <= distancia_max:
            return True
    return False


def amostrar_patches_pagina(frame_gray, boxes, patch_size, n_tentativas, rng):
    """
    Sorteia patches perto de texto real, mas so aceita os "quietos" (desvio
    baixo -- ver CLF_BG_MAX_STD). Isso deixa passar tanto fundo branco quanto
    fundo preto uniforme (ambos tem desvio baixo), e descarta o que tem muita
    variacao local (arte carregada, hachura densa, etc).
    """
    h, w = frame_gray.shape
    distancia_max = patch_size * CLF_BG_MAX_DIST_FACTOR
    patches = []
    for _ in range(n_tentativas):
        if w <= patch_size or h <= patch_size:
            break
        x = rng.randint(0, w - patch_size)
        y = rng.randint(0, h - patch_size)
        cx, cy = x + patch_size // 2, y + patch_size // 2
        if _sobrepoe(cx, cy, boxes, margem=patch_size // 2):
            continue
        if not _perto_de_algum_box(cx, cy, boxes, distancia_max):
            continue
        patch = frame_gray[y:y + patch_size, x:x + patch_size]
        if float(patch.std()) > CLF_BG_MAX_STD:
            continue
        patches.append(patch)
    return patches


def main():
    dir_claro = os.path.join(BACKGROUNDS_REAL_DIR, "claro")
    dir_escuro = os.path.join(BACKGROUNDS_REAL_DIR, "escuro")
    os.makedirs(dir_claro, exist_ok=True)
    os.makedirs(dir_escuro, exist_ok=True)
    rng = random.Random(42)

    print("[INFO] Carregando detector...")
    detector = load_detector(DETECTOR_WEIGHTS_PATH)

    paginas = listar_paginas(CLF_BG_HARVEST_PAGINAS, seed=42)
    print(f"[INFO] Varrendo ate {len(paginas)} paginas -- alvo: "
          f"{CLF_BG_HARVEST_COUNT} claros + {CLF_BG_HARVEST_COUNT_ESCURO} escuros")

    n_claro, n_escuro = 0, 0
    pbar = tqdm(paginas, desc="paginas")
    for vol, img_path in pbar:
        if n_claro >= CLF_BG_HARVEST_COUNT and n_escuro >= CLF_BG_HARVEST_COUNT_ESCURO:
            break
        frame = cv2.imread(img_path)
        if frame is None:
            continue
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        results = detector(
            frame, conf=PIPELINE_DET_CONF, iou=PIPELINE_DET_IOU,
            agnostic_nms=True, max_det=PIPELINE_DET_MAX_DET, verbose=False,
        )[0]
        boxes = [tuple(b.xyxy[0].tolist()) for b in results.boxes]

        patches = amostrar_patches_pagina(frame_gray, boxes, CLF_BG_PATCH_SIZE, n_tentativas=80, rng=rng)
        for patch in patches:
            escuro = float(patch.mean()) < CLF_BG_DARK_MEAN_THRESHOLD
            if escuro:
                if n_escuro >= CLF_BG_HARVEST_COUNT_ESCURO:
                    continue
                Image.fromarray(patch).save(os.path.join(dir_escuro, f"{n_escuro:05d}.png"))
                n_escuro += 1
            else:
                if n_claro >= CLF_BG_HARVEST_COUNT:
                    continue
                Image.fromarray(patch).save(os.path.join(dir_claro, f"{n_claro:05d}.png"))
                n_claro += 1
        pbar.set_postfix(claro=n_claro, escuro=n_escuro)

    print(f"\n[INFO] Claros: {n_claro} em {dir_claro}")
    print(f"[INFO] Escuros: {n_escuro} em {dir_escuro}")
    if n_claro < CLF_BG_HARVEST_COUNT or n_escuro < CLF_BG_HARVEST_COUNT_ESCURO:
        print(f"[AVISO] Nao bateu a meta varrendo {len(paginas)} paginas. "
              f"Aumente KD_CLF_BG_HARVEST_PAGINAS se precisar de mais.")


if __name__ == "__main__":
    main()
