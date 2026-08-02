"""
manga109_align_n1.py
=====================
Gera um conjunto de validação REAL de kanji N1 a partir do Manga109, pra usar
na seleção de checkpoint do classificador em vez de confiar só no `val_acc`
sintético (que não enxerga nada fora da distribuição do próprio gerador --
ver EXPERIMENTS.md "Regressão do checkpoint 2026-08-01" pra um caso real
onde isso mascarou uma regressão de recall real).

Reusa a mesma heurística de alinhamento de `manga109_align.py` (detector
próprio + transcrição oficial do Manga109 cruzados por contagem/coluna), mas
ao contrário dela GUARDA os caracteres N1 em vez de descartar. A diferença é
que aqui N1 é o alvo, então cada crop alinhado passa por uma checagem
adicional antes de ser aceito: o manga-ocr (kha-white/manga-ocr-base,
especializado em texto de mangá) precisa concordar com o rótulo.

Por quê essa checagem extra: a heurística de contagem/coluna tem um erro
residual real (auditoria visual encontrou ~6% de rótulo errado numa amostra
sem filtro -- um caractere perdido pelo detector + uma caixa espúria em
outro lugar da mesma linha cancelam na CONTAGEM total mas deslocam o
zip bbox<->char a partir do erro). Testado empiricamente (ver EXPERIMENTS.md):
  - EasyOCR/Tesseract como filtro: descartam 57% do dado BOM (são fracos
    demais nesse domínio -- é a própria conclusão da Hipótese 1 do projeto),
    inviável.
  - manga-ocr como filtro: ~24% de rejeição no total (mistura de erros reais
    pegos + alguns falsos positivos onde o manga-ocr erra um crop correto),
    mas auditoria visual do lado ACEITO não achou nenhum erro numa amostra
    de 64 -- ou seja, é conservador (perde algum dado bom) mas não deixa
    passar dado ruim, exatamente o trade-off certo pra um conjunto de
    validação (precisão importa mais que cobertura aqui).

Sem split train/val: esse conjunto nunca é usado pra treino (sem gradiente
fluindo dele), só pra avaliar checkpoints já treinados -- não existe risco
de vazamento a evitar.

Uso:
    python -m src.helper.manga109_align_n1                   # todos os volumes
    python -m src.helper.manga109_align_n1 --limit-volumes 5  # teste rapido
"""

import argparse
import os
import xml.etree.ElementTree as ET
from glob import glob

import cv2
from PIL import Image
from tqdm import tqdm

from config import (
    MANGA109_ANNOTATIONS, MANGA109_IMAGES, MANGA109_ALIGN_N1_DIR,
    DETECTOR_WEIGHTS_PATH, PIPELINE_DET_CONF, PIPELINE_DET_IOU, PIPELINE_DET_MAX_DET,
    PIPELINE_BBOX_PADDING,
)
from src.helper.kanjis import get_kanjis
from src.helper.manga109_align import alinhar_pagina
from src.pipeline.inference import load_detector, _expandir_bbox


def processar_volume(nome_volume: str, detector, n1_set: set, ler_mangaocr) -> dict:
    """Processa todas as paginas de um volume. Retorna contagem por categoria."""
    xml_path = os.path.join(MANGA109_ANNOTATIONS, f"{nome_volume}.xml")
    tree = ET.parse(xml_path)
    root = tree.getroot()

    stats = {"bruto": 0, "aceito": 0, "paginas": 0}

    for page in root.iter("page"):
        idx = int(page.get("index"))
        img_path = os.path.join(MANGA109_IMAGES, nome_volume, f"{idx:03d}.jpg")
        if not os.path.exists(img_path):
            continue

        frame = cv2.imread(img_path)
        if frame is None:
            continue
        h_frame, w_frame = frame.shape[:2]

        results = detector(
            frame, conf=PIPELINE_DET_CONF, iou=PIPELINE_DET_IOU,
            agnostic_nms=True, max_det=PIPELINE_DET_MAX_DET, verbose=False,
        )[0]
        todos_boxes = [tuple(b.xyxy[0].tolist()) for b in results.boxes]

        text_elements = list(page.iter("text"))
        pares = alinhar_pagina(frame, text_elements, todos_boxes)

        frame_rgb = Image.fromarray(frame[:, :, ::-1])
        for i, (box, char) in enumerate(pares):
            if char not in n1_set:
                continue

            x1, y1, x2, y2 = box
            xe1, ye1, xe2, ye2 = _expandir_bbox(
                x1, y1, x2, y2, PIPELINE_BBOX_PADDING, w_frame, h_frame
            )
            crop = frame_rgb.crop((xe1, ye1, xe2, ye2)).convert("L")
            stats["bruto"] += 1

            leitura = ler_mangaocr(crop.convert("RGB")).strip()
            if char not in leitura:
                continue

            codepoint = f"U+{ord(char):04X}"
            dest = os.path.join(MANGA109_ALIGN_N1_DIR, codepoint)
            os.makedirs(dest, exist_ok=True)
            crop.save(os.path.join(dest, f"{nome_volume}_{idx:03d}_{i:03d}.png"))
            stats["aceito"] += 1

        stats["paginas"] += 1

    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit-volumes", type=int, default=None,
                        help="Processa so os N primeiros volumes (para teste rapido)")
    args = parser.parse_args()

    print("[INFO] Carregando detector...")
    detector = load_detector(DETECTOR_WEIGHTS_PATH)
    n1_set = set(get_kanjis("n1"))

    print("[INFO] Carregando manga-ocr...")
    from manga_ocr import MangaOcr
    mocr = MangaOcr()
    ler_mangaocr = lambda img: mocr(img)

    volumes = sorted(
        os.path.splitext(os.path.basename(f))[0]
        for f in glob(os.path.join(MANGA109_ANNOTATIONS, "*.xml"))
    )
    if args.limit_volumes:
        volumes = volumes[:args.limit_volumes]

    print(f"[INFO] Processando {len(volumes)} volumes...")
    total = {"bruto": 0, "aceito": 0, "paginas": 0}
    for nome in tqdm(volumes, desc="volumes"):
        stats = processar_volume(nome, detector, n1_set, ler_mangaocr)
        for k in total:
            total[k] += stats[k]

    n_classes = len(os.listdir(MANGA109_ALIGN_N1_DIR)) if os.path.isdir(MANGA109_ALIGN_N1_DIR) else 0
    print(f"\n[INFO] Paginas processadas: {total['paginas']}")
    print(f"[INFO] Crops N1 brutos: {total['bruto']}")
    print(f"[INFO] Crops N1 aceitos (manga-ocr concordou): {total['aceito']} "
          f"({100 * total['aceito'] / max(1, total['bruto']):.1f}%)")
    print(f"[INFO] Classes distintas cobertas: {n_classes}")
    print(f"[INFO] Saida em: {MANGA109_ALIGN_N1_DIR}")


if __name__ == "__main__":
    main()
