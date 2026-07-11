"""
measure_real_stats.py
=====================
Mede estatísticas do dado REAL (manga de verdade, via Manga109 + nosso
detector) para calibrar o gerador sintético (`src/classifier/generate_crops.py`)
contra números medidos em vez de valores chutados.

Não precisa de rótulo nenhum — só roda o detector já treinado numa amostra
de páginas e mede, por bbox detectada:

  1. Tamanho real do crop (largura/altura em pixel, antes do resize pro
     tamanho de entrada do classificador) — pra saber se 64x64 é gargalo
     de resolução ou já é maior que o detalhe real disponível.
  2. Nitidez (variância do Laplaciano) do crop já redimensionado pro
     tamanho de entrada — pra calibrar o range de blur sintético
     (CLF_BLUR_SIGMA_MIN/MAX) contra o nível real de blur em produção.
  3. Textura de fundo real (patches sem nenhuma bbox por perto) — pra
     comparar contra o "papel liso" sintético (CLF_BG_VALUE_MIN/MAX +
     ruído gaussiano) e ver o quão diferente é da textura real
     (screentone, arte, etc).

Salva:
  - data/real_stats/tamanhos_bbox.csv        (largura, altura, blur_var por crop)
  - data/real_stats/fundo_stats.csv          (media, desvio por patch de fundo)
  - data/real_stats/contato_crops_reais.png  (grade de crops reais amostrados)
  - data/real_stats/contato_fundos_reais.png (grade de patches de fundo reais)
  - Resumo estatístico impresso no console (percentis).

Uso:
    python -m src.helper.measure_real_stats --n-paginas 60
"""

import argparse
import csv
import os
import random

import cv2
import numpy as np
from PIL import Image

from config import (
    MANGA109_ANNOTATIONS, MANGA109_IMAGES,
    DETECTOR_WEIGHTS_PATH, PIPELINE_DET_CONF, PIPELINE_DET_IOU, PIPELINE_DET_MAX_DET,
    PIPELINE_BBOX_PADDING, CLASSIFIER_INPUT_SIZE, ROOT_DIR,
)
from src.pipeline.inference import load_detector, _expandir_bbox

OUT_DIR = os.path.join(ROOT_DIR, "data", "real_stats")


# ---------------------------------------------------------------------------
# Amostragem de páginas
# ---------------------------------------------------------------------------

def listar_paginas(n_paginas: int, seed: int = 42) -> list:
    """
    Junta (volume, caminho_imagem) de todos os volumes disponiveis e sorteia
    uma amostra espalhada -- espalhar entre volumes da mais diversidade de
    fonte/estilo de traço do que pegar N paginas seguidas de um so volume.
    """
    volumes = sorted(
        os.path.splitext(f)[0] for f in os.listdir(MANGA109_ANNOTATIONS) if f.endswith(".xml")
    )
    todas = []
    for vol in volumes:
        vol_dir = os.path.join(MANGA109_IMAGES, vol)
        if not os.path.isdir(vol_dir):
            continue
        for fname in os.listdir(vol_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                todas.append((vol, os.path.join(vol_dir, fname)))

    rng = random.Random(seed)
    rng.shuffle(todas)
    return todas[:n_paginas]


# ---------------------------------------------------------------------------
# Medições
# ---------------------------------------------------------------------------

def blur_var(img_gray: np.ndarray) -> float:
    """Variancia do Laplaciano -- metrica padrao de nitidez (maior = mais nitido)."""
    return float(cv2.Laplacian(img_gray, cv2.CV_64F).var())


def _sobrepoe(px, py, boxes, margem):
    """True se o ponto (com margem) cai dentro ou perto de alguma bbox detectada."""
    for (x1, y1, x2, y2) in boxes:
        if (x1 - margem) <= px <= (x2 + margem) and (y1 - margem) <= py <= (y2 + margem):
            return True
    return False


def amostrar_fundo(frame_gray, boxes, patch_size: int, n_tentativas: int, rng: random.Random) -> list:
    """Sorteia patches quadrados que nao encostam em nenhuma bbox detectada."""
    h, w = frame_gray.shape
    patches = []
    for _ in range(n_tentativas):
        if w <= patch_size or h <= patch_size:
            break
        x = rng.randint(0, w - patch_size)
        y = rng.randint(0, h - patch_size)
        cx, cy = x + patch_size // 2, y + patch_size // 2
        if _sobrepoe(cx, cy, boxes, margem=patch_size // 2):
            continue
        patches.append(frame_gray[y:y + patch_size, x:x + patch_size])
    return patches


# ---------------------------------------------------------------------------
# Contact sheet (grade de imagens pra inspecao visual)
# ---------------------------------------------------------------------------

def salvar_contact_sheet(imgs: list, path: str, cols: int = 10, cell: int = 64):
    if not imgs:
        print(f"[AVISO] Nada pra salvar em {path}")
        return
    rows = (len(imgs) + cols - 1) // cols
    sheet = np.full((rows * cell, cols * cell), 255, dtype=np.uint8)
    for i, img in enumerate(imgs):
        r, c = divmod(i, cols)
        resized = cv2.resize(img, (cell, cell))
        sheet[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell] = resized
    Image.fromarray(sheet).save(path)
    print(f"[INFO] Contact sheet salvo em: {path}")


# ---------------------------------------------------------------------------
# Loop principal
# ---------------------------------------------------------------------------

def percentis(valores: list) -> dict:
    arr = np.array(valores)
    return {p: float(np.percentile(arr, p)) for p in (5, 25, 50, 75, 95)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-paginas", type=int, default=60,
                        help="Quantas paginas amostrar (espalhadas entre volumes)")
    parser.add_argument("--max-crops-sheet", type=int, default=100,
                        help="Quantos crops reais colocar no contact sheet")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    rng = random.Random(args.seed)

    print("[INFO] Carregando detector...")
    detector = load_detector(DETECTOR_WEIGHTS_PATH)

    paginas = listar_paginas(args.n_paginas, seed=args.seed)
    print(f"[INFO] Amostrando {len(paginas)} paginas (espalhadas entre volumes)...")

    linhas_bbox = []       # (volume, largura, altura, blur_var_resized)
    linhas_fundo = []      # (volume, media, desvio)
    crops_para_sheet = []
    fundos_para_sheet = []

    for vol, img_path in paginas:
        frame = cv2.imread(img_path)
        if frame is None:
            continue
        h_frame, w_frame = frame.shape[:2]
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        results = detector(
            frame, conf=PIPELINE_DET_CONF, iou=PIPELINE_DET_IOU,
            agnostic_nms=True, max_det=PIPELINE_DET_MAX_DET, verbose=False,
        )[0]
        boxes = [tuple(b.xyxy[0].tolist()) for b in results.boxes]

        for (x1, y1, x2, y2) in boxes:
            largura, altura = x2 - x1, y2 - y1
            if largura <= 1 or altura <= 1:
                continue

            xe1, ye1, xe2, ye2 = _expandir_bbox(
                x1, y1, x2, y2, PIPELINE_BBOX_PADDING, w_frame, h_frame
            )
            crop = frame_gray[int(ye1):int(ye2), int(xe1):int(xe2)]
            if crop.size == 0:
                continue
            crop_resized = cv2.resize(crop, (CLASSIFIER_INPUT_SIZE, CLASSIFIER_INPUT_SIZE))

            linhas_bbox.append((vol, largura, altura, blur_var(crop_resized)))
            if len(crops_para_sheet) < args.max_crops_sheet and rng.random() < 0.3:
                crops_para_sheet.append(crop_resized)

        # Fundo: usa o tamanho mediano de bbox dessa pagina (ou o input size, se nao achou nada)
        if boxes:
            larguras = sorted(b[2] - b[0] for b in boxes)
            patch_size = max(16, int(larguras[len(larguras) // 2]))
        else:
            patch_size = CLASSIFIER_INPUT_SIZE

        fundos = amostrar_fundo(frame_gray, boxes, patch_size, n_tentativas=15, rng=rng)
        for patch in fundos:
            linhas_fundo.append((vol, float(patch.mean()), float(patch.std())))
            if len(fundos_para_sheet) < args.max_crops_sheet and rng.random() < 0.5:
                fundos_para_sheet.append(patch)

    if not linhas_bbox:
        print("[AVISO] Nenhuma bbox detectada na amostra -- nada pra medir.")
        return

    # --- CSV ---
    with open(os.path.join(OUT_DIR, "tamanhos_bbox.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["volume", "largura", "altura", "blur_var_resized"])
        w.writerows(linhas_bbox)

    with open(os.path.join(OUT_DIR, "fundo_stats.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["volume", "media", "desvio"])
        w.writerows(linhas_fundo)

    # --- Contact sheets ---
    salvar_contact_sheet(crops_para_sheet, os.path.join(OUT_DIR, "contato_crops_reais.png"))
    salvar_contact_sheet(fundos_para_sheet, os.path.join(OUT_DIR, "contato_fundos_reais.png"))

    # --- Resumo ---
    larguras = [l for _, l, a, b in linhas_bbox]
    alturas = [a for _, l, a, b in linhas_bbox]
    blurs = [b for _, l, a, b in linhas_bbox]
    medias_fundo = [m for _, m, d in linhas_fundo]
    desvios_fundo = [d for _, m, d in linhas_fundo]

    print(f"\n[INFO] Total de bboxes medidas: {len(linhas_bbox)} (de {len(paginas)} paginas)")
    print(f"[INFO] Total de patches de fundo: {len(linhas_fundo)}")

    print("\n=== Tamanho real da bbox (pixels, antes do resize) ===")
    print("Largura  p5/p25/p50/p75/p95:", percentis(larguras))
    print("Altura   p5/p25/p50/p75/p95:", percentis(alturas))

    print("\n=== Nitidez (variancia do Laplaciano, apos resize p/ CLASSIFIER_INPUT_SIZE) ===")
    print("Blur_var p5/p25/p50/p75/p95:", percentis(blurs))
    print("(Quanto MENOR, mais borrado. Compare com o range que CLF_BLUR_SIGMA produz sinteticamente.)")

    print("\n=== Fundo real (patches sem bbox por perto) ===")
    print("Media    p5/p25/p50/p75/p95:", percentis(medias_fundo))
    print("Desvio   p5/p25/p50/p75/p95:", percentis(desvios_fundo))
    print("(Compare com CLF_BG_VALUE_MIN/MAX=hoje fixo e CLF_BG_NOISE_STD=desvio sintetico hoje.)")

    print(f"\n[INFO] CSVs e contact sheets salvos em: {OUT_DIR}")


if __name__ == "__main__":
    main()
