"""
detector_fp_check.py
=====================
Mede detecções do detector em páginas do Manga109 que não têm NENHUM <text>
anotado -- proxy barato de "sem caractere", mas IMPERFEITO: essas páginas
ainda podem ter efeito sonoro estilizado, logo ou título desenhados como
parte da arte (não capturados pela tag <text> do Manga109, ver
src/detector/synth_page.py), que são character-shaped de verdade e uma
detecção correta neles não é bug. Por isso o número absoluto não deve ser
lido como "taxa de falso positivo real" -- o uso pretendido é COMPARATIVO:
rodar o mesmo conjunto de páginas contra dois checkpoints (antigo vs. novo)
e comparar a contagem lado a lado. Se o checkpoint novo (treinado com
sintético, ver src/detector/generate_pages.py) detectar visivelmente MAIS
que o antigo nessas páginas, é sinal de que o sintético piorou a precisão
("detectar qualquer coisa") -- vira critério objetivo de aceite/rejeição.

Uso:
    python -m src.helper.detector_fp_check --pesos weights/best.pt
    python -m src.helper.detector_fp_check --pesos weights/best.pt weights/best_novo.pt
"""

import argparse
import os
import xml.etree.ElementTree as ET
from glob import glob

import cv2
from tqdm import tqdm

from config import MANGA109_ANNOTATIONS, MANGA109_IMAGES, PIPELINE_DET_CONF, PIPELINE_DET_IOU, PIPELINE_DET_MAX_DET
from src.pipeline.inference import load_detector


def listar_paginas_sem_texto(limit: int = None) -> list:
    """(volume, indice, img_path) de toda pagina cujo XML nao tem nenhum <text>."""
    volumes = sorted(
        os.path.splitext(os.path.basename(f))[0]
        for f in glob(os.path.join(MANGA109_ANNOTATIONS, "*.xml"))
    )
    paginas = []
    for volume in volumes:
        root = ET.parse(os.path.join(MANGA109_ANNOTATIONS, f"{volume}.xml")).getroot()
        for page in root.iter("page"):
            if list(page.iter("text")):
                continue
            idx = int(page.get("index"))
            img_path = os.path.join(MANGA109_IMAGES, volume, f"{idx:03d}.jpg")
            if os.path.exists(img_path):
                paginas.append((volume, idx, img_path))
    if limit:
        paginas = paginas[:limit]
    return paginas


def contar_deteccoes(weights_path: str, paginas: list) -> dict:
    detector = load_detector(weights_path)
    total_deteccoes, paginas_com_deteccao = 0, 0

    for _, _, img_path in tqdm(paginas, desc=os.path.basename(weights_path)):
        frame = cv2.imread(img_path)
        if frame is None:
            continue
        results = detector(
            frame, conf=PIPELINE_DET_CONF, iou=PIPELINE_DET_IOU,
            agnostic_nms=True, max_det=PIPELINE_DET_MAX_DET, verbose=False,
        )[0]
        n = len(results.boxes)
        total_deteccoes += n
        if n > 0:
            paginas_com_deteccao += 1

    n_paginas = len(paginas)
    return {
        "pesos": weights_path,
        "n_paginas": n_paginas,
        "total_deteccoes": total_deteccoes,
        "deteccoes_por_pagina": total_deteccoes / max(1, n_paginas),
        "paginas_com_deteccao": paginas_com_deteccao,
        "pct_paginas_com_deteccao": 100 * paginas_com_deteccao / max(1, n_paginas),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pesos", nargs="+", required=True,
                        help="1 ou mais caminhos de peso do detector, pra comparar lado a lado")
    parser.add_argument("--limit-paginas", type=int, default=200)
    args = parser.parse_args()

    print("[INFO] Listando paginas sem nenhum <text> anotado...")
    paginas = listar_paginas_sem_texto(limit=args.limit_paginas)
    print(f"[INFO] {len(paginas)} paginas -- LEMBRETE: proxy imperfeito de \"sem "
          f"caractere\" (SFX/logo/titulo desenhados como arte nao contam como "
          f"<text> mas sao character-shaped de verdade). Leia o numero de forma "
          f"COMPARATIVA entre checkpoints, nao como taxa de falso positivo absoluta.")

    resultados = [contar_deteccoes(p, paginas) for p in args.pesos]

    print(f"\n{'pesos':<40} {'deteccoes/pagina':>18} {'% paginas c/ deteccao':>22}")
    for r in resultados:
        print(f"{os.path.basename(r['pesos']):<40} {r['deteccoes_por_pagina']:>18.3f} "
              f"{r['pct_paginas_com_deteccao']:>21.1f}%")


if __name__ == "__main__":
    main()
