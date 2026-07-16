"""
validacao_manual.py
====================
Gera, para cada página em data/benchmark/ground_truth.json, uma imagem
anotada salva em data/validacao_manual/ com as detecções do pipeline
coloridas por resultado, pra inspeção visual rápida do progresso do projeto
(sem precisar reler números de recall soltos).

Detecções classificadas como OUTROS não são desenhadas -- só interessam as
que o classificador arriscou como algum kanji N1 real. Cor por detecção
(comparada contra a linha oficial do Manga109 em que o centro da detecção
cai, se houver):
  - verde:   kanji previsto está na lista de N1 esperados da linha (acerto)
  - laranja: previu algum kanji que não está nos N1 esperados da linha
             (classe errada)
  - cinza:   centro da detecção fora de qualquer linha anotada (furigana,
             outro texto da página, etc. -- fora do escopo verificável)

As bboxes das linhas oficiais são desenhadas em azul fino como referência.

Uso:
    python -m src.helper.validacao_manual
"""

import glob
import json
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from config import ROOT_DIR, FONTS_DIR
from src.pipeline.inference import Pipeline
from src.helper.manga109_align import _dentro

BENCHMARK_DIR = os.path.join(ROOT_DIR, "data", "benchmark")
GT_PATH = os.path.join(BENCHMARK_DIR, "ground_truth.json")
OUT_DIR = os.path.join(ROOT_DIR, "data", "validacao_manual")

VERDE = (0, 200, 0)
LARANJA = (255, 140, 0)
CINZA = (140, 140, 140)
AZUL_LINHA = (0, 120, 255)


def carregar_fonte(tamanho=20):
    candidatos = glob.glob(os.path.join(FONTS_DIR, "*.ttf"))
    if not candidatos:
        raise FileNotFoundError(
            "Nenhuma fonte encontrada em assets/fonts/. Rode `python -m src.helper.fonts` primeiro."
        )
    return ImageFont.truetype(candidatos[0], tamanho)


def cor_deteccao(det, linhas):
    for linha in linhas:
        x1, y1, x2, y2 = linha["bbox"]
        if _dentro(det.bbox, x1, y1, x2, y2):
            return VERDE if det.kanji in linha["n1_esperados"] else LARANJA
    return CINZA


def anotar_pagina(frame_bgr, deteccoes, linhas, fonte):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(img_pil)

    for linha in linhas:
        draw.rectangle(linha["bbox"], outline=AZUL_LINHA, width=1)

    for det in deteccoes:
        if det.codepoint == "OUTROS":
            continue
        cor = cor_deteccao(det, linhas)
        x1, y1, x2, y2 = det.bbox
        draw.rectangle([x1, y1, x2, y2], outline=cor, width=2)
        draw.text((x1, max(0, y1 - 22)), det.kanji, font=fonte, fill=cor)

    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    with open(GT_PATH, encoding="utf-8") as f:
        gt = json.load(f)

    print("[INFO] Carregando pipeline...")
    pipeline = Pipeline()
    fonte = carregar_fonte()

    for pagina in gt["paginas"]:
        img_path = os.path.join(BENCHMARK_DIR, pagina["imagem"])
        frame = cv2.imread(img_path)
        if frame is None:
            raise FileNotFoundError(f"Nao foi possivel ler {img_path}")

        origem = pagina["origem"]
        print(f"[INFO] Anotando {pagina['imagem']} ({origem['volume']} pag {origem['pagina']})...")
        deteccoes = pipeline.predict(frame)
        anotada = anotar_pagina(frame, deteccoes, pagina["linhas"], fonte)

        nome_saida = f"{origem['volume']}_{origem['pagina']}.jpg"
        out_path = os.path.join(OUT_DIR, nome_saida)
        cv2.imwrite(out_path, anotada)
        print(f"  -> salvo em {out_path}")

    print(f"\n[INFO] Concluido. Imagens em {OUT_DIR}")


if __name__ == "__main__":
    main()
