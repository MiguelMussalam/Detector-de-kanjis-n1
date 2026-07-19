"""
generate_pages.py
==================
CLI que gera o dataset sintético de páginas pro detector (imagens + labels
YOLO, classe única "glifo"), ancorando cada glifo sintético numa bbox de
linha <text> oficial do Manga109 (ver src/detector/synth_page.py pra lógica
de composição por página).

Exclui os volumes de MANGA109_ALIGN_VAL_VOLUMES do pool de páginas-fonte --
mesma convenção já usada pro classificador, evita vazar nas validações
full-corpus (src/helper/corpus_validate.py).

Sempre termina gerando um contact sheet (grade de páginas com bbox
desenhada) -- gate obrigatório de auditoria visual antes de qualquer treino
real em cima desse dado (principal risco: qualidade do inpaint em fundos
texturizados).

Uso:
    python -m src.detector.generate_pages [--n-pages N] [--out DIR]
"""

import argparse
import csv
import os
import random
import xml.etree.ElementTree as ET
from glob import glob

import cv2
import numpy as np
from tqdm import tqdm

from config import (
    MANGA109_ANNOTATIONS, MANGA109_IMAGES, MANGA109_ALIGN_VAL_VOLUMES,
    DETSYN_OUTPUT_DIR, DETSYN_N_PAGES, DETSYN_PROB_PAGINA_SEM_TEXTO,
)
from src.helper.kanjis import get_kanjis
from src.helper.fonts import get_fonts_list
from src.detector.synth_page import processar_pagina_para_synth, bbox_para_yolo


def _listar_paginas_candidatas(seed=42):
    """
    Retorna duas listas: páginas COM <text> (fonte pra síntese) e páginas SEM
    nenhum <text> (fonte pra negativo puro -- imagem copiada como está, label
    vazio), excluindo os volumes reservados pra validação.
    """
    volumes = sorted(
        os.path.splitext(os.path.basename(f))[0]
        for f in glob(os.path.join(MANGA109_ANNOTATIONS, "*.xml"))
        if os.path.splitext(os.path.basename(f))[0] not in MANGA109_ALIGN_VAL_VOLUMES
    )

    com_texto, sem_texto = [], []
    for volume in volumes:
        root = ET.parse(os.path.join(MANGA109_ANNOTATIONS, f"{volume}.xml")).getroot()
        for page in root.iter("page"):
            idx = int(page.get("index"))
            img_path = os.path.join(MANGA109_IMAGES, volume, f"{idx:03d}.jpg")
            if not os.path.exists(img_path):
                continue
            text_elements = list(page.iter("text"))
            if text_elements:
                com_texto.append((volume, idx, img_path, text_elements))
            else:
                sem_texto.append((volume, idx, img_path))

    rng = random.Random(seed)
    rng.shuffle(com_texto)
    rng.shuffle(sem_texto)
    return com_texto, sem_texto


def _salvar_contact_sheet(img_dir: str, lbl_dir: str, nomes: list, out_path: str,
                          cols: int = 4, cell: int = 500):
    """Grade de páginas com bbox desenhada (verde), pra auditoria visual --
    principal risco: qualidade do inpaint em fundo texturizado."""
    if not nomes:
        print(f"[AVISO] Nada pra salvar em {out_path}")
        return

    rows = (len(nomes) + cols - 1) // cols
    sheet = np.full((rows * cell, cols * cell, 3), 255, dtype=np.uint8)

    for i, nome in enumerate(nomes):
        img_path = os.path.join(img_dir, f"{nome}.jpg")
        lbl_path = os.path.join(lbl_dir, f"{nome}.txt")
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]

        if os.path.exists(lbl_path):
            with open(lbl_path, encoding="utf-8") as f:
                for linha in f:
                    partes = linha.split()
                    if len(partes) != 5:
                        continue
                    _, cx, cy, bw, bh = map(float, partes)
                    x1 = int((cx - bw / 2) * w)
                    y1 = int((cy - bh / 2) * h)
                    x2 = int((cx + bw / 2) * w)
                    y2 = int((cy + bh / 2) * h)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0), 1)

        thumb = cv2.resize(img, (cell, cell))
        r, c = divmod(i, cols)
        sheet[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell] = thumb

    cv2.imwrite(out_path, sheet)
    print(f"[INFO] Contact sheet salvo em: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-pages", type=int, default=DETSYN_N_PAGES)
    parser.add_argument("--out", type=str, default=DETSYN_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--contact-sheet-n", type=int, default=20)
    args = parser.parse_args()

    img_dir = os.path.join(args.out, "images")
    lbl_dir = os.path.join(args.out, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    print("[INFO] Carregando kanji N1 e fontes...")
    n1_kanjis = get_kanjis("n1")
    fonts = get_fonts_list()
    if not fonts:
        raise FileNotFoundError(
            "Nenhuma fonte encontrada em assets/fonts/. Rode `python -m src.helper.fonts` primeiro."
        )

    print("[INFO] Listando paginas candidatas (excluindo volumes de validacao)...")
    com_texto, sem_texto = _listar_paginas_candidatas(seed=args.seed)
    print(f"[INFO] {len(com_texto)} paginas com texto, {len(sem_texto)} sem texto")
    if not com_texto and not sem_texto:
        raise FileNotFoundError(
            f"Nenhum volume/pagina encontrado em MANGA109_ANNOTATIONS={MANGA109_ANNOTATIONS} "
            f"nem MANGA109_IMAGES={MANGA109_IMAGES}. Confira se o dataset completo do Manga109 "
            f"(imagens + XML de anotacao <text>, nao so o dataset Roboflow de bbox) esta "
            f"anexado e se config.py resolveu o diretorio certo (ver print acima/[INFO ou AVISO] "
            f"'Manga109 dir...' no inicio do log)."
        )

    rng = random.Random(args.seed)
    n_negativas = min(round(args.n_pages * DETSYN_PROB_PAGINA_SEM_TEXTO), len(sem_texto))
    n_sintese = min(args.n_pages - n_negativas, len(com_texto))

    nomes_gerados = []
    manifest_path = os.path.join(args.out, "manifest.csv")
    with open(manifest_path, "w", newline="", encoding="utf-8") as f_manifest:
        writer = csv.writer(f_manifest)
        writer.writerow(["arquivo", "volume", "pagina", "tipo", "n_linhas_originais", "n_glifos"])

        print(f"[INFO] Gerando {n_sintese} paginas sintetizadas...")
        for volume, idx, img_path, text_elements in tqdm(com_texto[:n_sintese], desc="sintese"):
            frame = cv2.imread(img_path)
            if frame is None:
                continue
            h, w = frame.shape[:2]

            resultado, bboxes = processar_pagina_para_synth(frame, text_elements, n1_kanjis, fonts, rng)

            nome = f"{volume}_{idx:03d}"
            cv2.imwrite(os.path.join(img_dir, f"{nome}.jpg"), resultado)
            with open(os.path.join(lbl_dir, f"{nome}.txt"), "w", encoding="utf-8") as f_lbl:
                for bbox in bboxes:
                    cx, cy, bw, bh = bbox_para_yolo(bbox, w, h)
                    f_lbl.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

            writer.writerow([f"{nome}.jpg", volume, idx, "sintese", len(text_elements), len(bboxes)])
            nomes_gerados.append(nome)

        print(f"[INFO] Gerando {n_negativas} paginas negativas (sem texto)...")
        for volume, idx, img_path in tqdm(sem_texto[:n_negativas], desc="negativas"):
            frame = cv2.imread(img_path)
            if frame is None:
                continue
            nome = f"{volume}_{idx:03d}"
            cv2.imwrite(os.path.join(img_dir, f"{nome}.jpg"), frame)
            open(os.path.join(lbl_dir, f"{nome}.txt"), "w").close()  # label vazio

            writer.writerow([f"{nome}.jpg", volume, idx, "negativa", 0, 0])

    print(f"[INFO] Dataset sintetico em: {args.out}")
    print(f"[INFO] Manifest em: {manifest_path}")

    amostra = random.Random(args.seed).sample(nomes_gerados, min(args.contact_sheet_n, len(nomes_gerados)))
    _salvar_contact_sheet(img_dir, lbl_dir, amostra, os.path.join(args.out, "contact_sheet.jpg"))


if __name__ == "__main__":
    main()
