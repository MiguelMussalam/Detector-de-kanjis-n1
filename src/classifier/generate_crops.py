"""
generate_crops.py
=================
Gera dataset sintético de crops de kanji para treinar o classificador.

Para cada kanji da lista N1, renderiza N amostras (40 treino + 10 val por padrão)
combinando aleatoriamente:
  - Fonte (das disponíveis em assets/fonts/)
  - Tamanho de fonte simulado (multi-scale)
  - Degradações (translate, morfologia, rotação, blur, ruído, contraste, JPEG)

Estrutura de output:
    data/classifier/
    ├── train/U+XXXX/NNN.png
    ├── val/U+XXXX/NNN.png
    └── sanity/NNN.png

Uso:
    # Geração completa
    python -m src.classifier.generate_crops

    # Geração de teste (só primeiras N classes)
    python -m src.classifier.generate_crops --limit 10

    # Só regenera modo sanity (rápido, pra iterar em degradações)
    python -m src.classifier.generate_crops --sanity-only
"""

import os
import io
import sys
import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from scipy.ndimage import grey_dilation, grey_erosion
from tqdm import tqdm

from config import (
    CLASSIFIER_TRAIN_DIR, CLASSIFIER_VAL_DIR, CLASSIFIER_SANITY_DIR,
    CLASSIFIER_KANJI_LEVEL, CLASSIFIER_INPUT_SIZE,
    CLASSIFIER_SAMPLES_TRAIN, CLASSIFIER_SAMPLES_VAL, CLASSIFIER_SANITY_COUNT,
    CLASSIFIER_SEED_TRAIN, CLASSIFIER_SEED_VAL,
    CLASSIFIER_FONT_SIZES, CLASSIFIER_CANVAS_MARGIN,
    CLF_BG_VALUE_MIN, CLF_BG_VALUE_MAX, CLF_BG_NOISE_STD,
    CLF_TRANSLATE_PROB, CLF_TRANSLATE_MAX,
    CLF_BLUR_PROB, CLF_BLUR_SIGMA_MIN, CLF_BLUR_SIGMA_MAX,
    CLF_NOISE_PROB, CLF_NOISE_STD_MIN, CLF_NOISE_STD_MAX,
    CLF_BRIGHTNESS_PROB, CLF_BRIGHTNESS_RANGE,
    CLF_CONTRAST_PROB, CLF_CONTRAST_RANGE,
    CLF_MORFO_PROB, CLF_MORFO_K_MIN, CLF_MORFO_K_MAX,
    CLF_JPEG_PROB, CLF_JPEG_QUALITY_MIN, CLF_JPEG_QUALITY_MAX,
    CLF_ROTATION_PROB, CLF_ROTATION_MAX,
)
from src.helper.kanjis import get_kanjis
from src.helper.fonts import get_fonts_list


# ---------------------------------------------------------------------------
# Renderização base
# ---------------------------------------------------------------------------

def render_glyph(char: str, font_path: str, font_size: int,
                 canvas_size: int) -> np.ndarray:
    """
    Renderiza um caractere em canvas quadrado, glyph centralizado, fundo branco.
    Retorna array uint8 (H, W) em grayscale.
    """
    font = ImageFont.truetype(font_path, size=font_size)
    canvas = Image.new("L", (canvas_size, canvas_size), color=255)
    draw = ImageDraw.Draw(canvas)

    # Bbox do glyph nesse font size
    bbox = draw.textbbox((0, 0), char, font=font)
    glyph_w = bbox[2] - bbox[0]
    glyph_h = bbox[3] - bbox[1]

    # Centraliza no canvas (compensando offset do bbox)
    x = (canvas_size - glyph_w) // 2 - bbox[0]
    y = (canvas_size - glyph_h) // 2 - bbox[1]

    draw.text((x, y), char, font=font, fill=0)
    return np.array(canvas, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Degradações (aplicadas na ordem definida em docs)
# ---------------------------------------------------------------------------

def apply_morphology(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """Dilate ou erode leve com kernel pequeno."""
    if rng.random() > CLF_MORFO_PROB:
        return img
    k = rng.randint(CLF_MORFO_K_MIN, CLF_MORFO_K_MAX)
    op = rng.choice(["dilate", "erode"])
    # Nota: como glyph é preto (0) sobre branco (255), dilate expande fundo (traço mais fino)
    # e erode expande escuro (traço mais grosso). Trocamos os nomes para intuição correta.
    if op == "dilate":  # queremos traço mais grosso
        return grey_erosion(img, size=(k, k))
    else:               # queremos traço mais fino
        return grey_dilation(img, size=(k, k))


def apply_rotation(img: np.ndarray, rng: random.Random) -> np.ndarray:
    if rng.random() > CLF_ROTATION_PROB:
        return img
    angle = rng.uniform(-CLF_ROTATION_MAX, CLF_ROTATION_MAX)
    pil = Image.fromarray(img)
    pil = pil.rotate(angle, resample=Image.BILINEAR, fillcolor=255)
    return np.array(pil)


def apply_translate_and_crop(img: np.ndarray, rng: random.Random,
                             output_size: int) -> np.ndarray:
    """
    Aplica deslocamento aleatório e recorta para o tamanho final.
    Simula bbox imperfeita do detector.
    """
    h, w = img.shape
    if rng.random() > CLF_TRANSLATE_PROB:
        dx, dy = 0, 0
    else:
        max_dx = int(w * CLF_TRANSLATE_MAX)
        max_dy = int(h * CLF_TRANSLATE_MAX)
        dx = rng.randint(-max_dx, max_dx)
        dy = rng.randint(-max_dy, max_dy)

    # Centro de crop deslocado
    cx = w // 2 + dx
    cy = h // 2 + dy

    # Recorte quadrado ao redor do centro (dimensão baseada na maior)
    # Aqui só centraliza pra downsample no próximo passo
    crop_size = min(h, w)
    x0 = max(0, cx - crop_size // 2)
    y0 = max(0, cy - crop_size // 2)
    x1 = min(w, x0 + crop_size)
    y1 = min(h, y0 + crop_size)

    return img[y0:y1, x0:x1]


def resize_to_target(img: np.ndarray, target: int) -> np.ndarray:
    pil = Image.fromarray(img)
    pil = pil.resize((target, target), resample=Image.BILINEAR)
    return np.array(pil)


def apply_blur(img: np.ndarray, rng: random.Random) -> np.ndarray:
    if rng.random() > CLF_BLUR_PROB:
        return img
    sigma = rng.uniform(CLF_BLUR_SIGMA_MIN, CLF_BLUR_SIGMA_MAX)
    pil = Image.fromarray(img).filter(ImageFilter.GaussianBlur(radius=sigma))
    return np.array(pil)


def apply_noise(img: np.ndarray, rng: random.Random) -> np.ndarray:
    if rng.random() > CLF_NOISE_PROB:
        return img
    std = rng.uniform(CLF_NOISE_STD_MIN, CLF_NOISE_STD_MAX) * 255
    noise = np.random.normal(0, std, img.shape)
    out = img.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_brightness_contrast(img: np.ndarray, rng: random.Random) -> np.ndarray:
    out = img.astype(np.float32)
    if rng.random() < CLF_BRIGHTNESS_PROB:
        delta = rng.uniform(-CLF_BRIGHTNESS_RANGE, CLF_BRIGHTNESS_RANGE) * 255
        out = out + delta
    if rng.random() < CLF_CONTRAST_PROB:
        factor = 1.0 + rng.uniform(-CLF_CONTRAST_RANGE, CLF_CONTRAST_RANGE)
        mean = out.mean()
        out = (out - mean) * factor + mean
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_jpeg(img: np.ndarray, rng: random.Random) -> np.ndarray:
    if rng.random() > CLF_JPEG_PROB:
        return img
    q = rng.randint(CLF_JPEG_QUALITY_MIN, CLF_JPEG_QUALITY_MAX)
    pil = Image.fromarray(img)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=q)
    buf.seek(0)
    return np.array(Image.open(buf))


def apply_paper_background(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """
    Substitui fundo branco puro por fundo de papel: valor levemente amarelado
    entre CLF_BG_VALUE_MIN e MAX, com ruído gaussiano leve.
    Areas onde o glyph existe (pixels escuros) são preservadas.
    """
    h, w = img.shape
    bg_value = rng.uniform(CLF_BG_VALUE_MIN, CLF_BG_VALUE_MAX)
    bg = np.full((h, w), bg_value, dtype=np.float32)
    noise = np.random.normal(0, CLF_BG_NOISE_STD, (h, w))
    bg = np.clip(bg + noise, 0, 255).astype(np.uint8)

    # Mistura: onde img é branco (~255), usa bg; onde tem glyph (escuro), preserva
    mask = img > 240  # threshold — pixels quase-brancos vão pro fundo
    out = img.copy()
    out[mask] = bg[mask]
    return out


# ---------------------------------------------------------------------------
# Pipeline completo de uma amostra
# ---------------------------------------------------------------------------

def generate_sample(char: str, font_path: str, font_size: int,
                    rng: random.Random, output_size: int,
                    canvas_margin: float) -> np.ndarray:
    """
    Gera um crop 64x64 do kanji com degradações aplicadas em ordem.
    """
    # Canvas maior que o glyph pra dar espaço pras degradações
    canvas_size = int(font_size * (1 + 2 * canvas_margin))

    # 1. Render limpo
    img = render_glyph(char, font_path, font_size, canvas_size)

    # 2. Morfologia
    img = apply_morphology(img, rng)

    # 3. Rotação
    img = apply_rotation(img, rng)

    # 4. Translate + crop pro tamanho quadrado
    img = apply_translate_and_crop(img, rng, output_size)

    # 5. Resize pro tamanho final
    img = resize_to_target(img, output_size)

    # 6. Fundo de papel
    img = apply_paper_background(img, rng)

    # 7. Blur
    img = apply_blur(img, rng)

    # 8. Ruído
    img = apply_noise(img, rng)

    # 9. Contraste/brilho
    img = apply_brightness_contrast(img, rng)

    # 10. JPEG
    img = apply_jpeg(img, rng)

    return img


# ---------------------------------------------------------------------------
# Loop principal
# ---------------------------------------------------------------------------

def codepoint_dir(char: str) -> str:
    """Retorna nome de pasta no formato U+XXXX."""
    return f"U+{ord(char):04X}"


def generate_split(kanjis: list, fonts: list, samples_per_class: int,
                   output_dir: str, seed: int, split_name: str):
    """Gera um split (train ou val) inteiro."""
    rng = random.Random(seed)
    # Também setar np.random pra reprodutibilidade de np.random.normal
    np.random.seed(seed)

    os.makedirs(output_dir, exist_ok=True)

    for char in tqdm(kanjis, desc=f"[{split_name}]"):
        class_dir = os.path.join(output_dir, codepoint_dir(char))
        os.makedirs(class_dir, exist_ok=True)

        for i in range(samples_per_class):
            font_path = rng.choice(fonts)
            font_size = rng.choice(CLASSIFIER_FONT_SIZES)
            try:
                img = generate_sample(
                    char=char,
                    font_path=font_path,
                    font_size=font_size,
                    rng=rng,
                    output_size=CLASSIFIER_INPUT_SIZE,
                    canvas_margin=CLASSIFIER_CANVAS_MARGIN,
                )
                Image.fromarray(img).save(
                    os.path.join(class_dir, f"{i:03d}.png")
                )
            except Exception as e:
                print(f"[WARN] Falha em {char} com {os.path.basename(font_path)} @ {font_size}px: {e}")


def generate_sanity(kanjis: list, fonts: list, output_dir: str, count: int, seed: int):
    """Gera N amostras aleatórias pra inspeção visual (mix de classes)."""
    rng = random.Random(seed)
    np.random.seed(seed)

    os.makedirs(output_dir, exist_ok=True)

    for i in range(count):
        char = rng.choice(kanjis)
        font_path = rng.choice(fonts)
        font_size = rng.choice(CLASSIFIER_FONT_SIZES)
        try:
            img = generate_sample(
                char=char,
                font_path=font_path,
                font_size=font_size,
                rng=rng,
                output_size=CLASSIFIER_INPUT_SIZE,
                canvas_margin=CLASSIFIER_CANVAS_MARGIN,
            )
            filename = f"{i:03d}_{codepoint_dir(char)}_{os.path.basename(font_path)[:15]}_{font_size}.png"
            Image.fromarray(img).save(os.path.join(output_dir, filename))
        except Exception as e:
            print(f"[WARN] Falha sanity em {char}: {e}")

    print(f"[INFO] {count} amostras sanity em: {output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="Gera só as primeiras N classes (para testar rápido)")
    parser.add_argument("--sanity-only", action="store_true",
                        help="Só regenera o modo sanity, sem train/val")
    parser.add_argument("--skip-sanity", action="store_true",
                        help="Pula geração do sanity")
    args = parser.parse_args()

    # Carregar kanji e fontes
    print(f"[INFO] Carregando kanji nível {CLASSIFIER_KANJI_LEVEL}...")
    kanjis = get_kanjis(CLASSIFIER_KANJI_LEVEL)
    print(f"[INFO] Total: {len(kanjis)} kanji")

    if args.limit:
        kanjis = kanjis[:args.limit]
        print(f"[INFO] Limitado a {len(kanjis)} kanji (modo teste)")

    fonts = get_fonts_list()
    if not fonts:
        raise FileNotFoundError(
            "Nenhuma fonte encontrada em assets/fonts/. Rode `python -m src.helper.fonts` primeiro."
        )
    print(f"[INFO] Fontes: {len(fonts)}")
    for f in fonts:
        print(f"        - {os.path.basename(f)}")

    # Sanity
    if not args.skip_sanity:
        print(f"\n[INFO] Gerando {CLASSIFIER_SANITY_COUNT} amostras sanity...")
        generate_sanity(kanjis, fonts, CLASSIFIER_SANITY_DIR,
                        CLASSIFIER_SANITY_COUNT, CLASSIFIER_SEED_TRAIN)

    if args.sanity_only:
        print("[INFO] Modo sanity-only. Encerrando.")
        return

    # Train
    print(f"\n[INFO] Gerando treino: {CLASSIFIER_SAMPLES_TRAIN} amostras × {len(kanjis)} classes = "
          f"{CLASSIFIER_SAMPLES_TRAIN * len(kanjis)} crops")
    generate_split(kanjis, fonts, CLASSIFIER_SAMPLES_TRAIN,
                   CLASSIFIER_TRAIN_DIR, CLASSIFIER_SEED_TRAIN, "train")

    # Val
    print(f"\n[INFO] Gerando validação: {CLASSIFIER_SAMPLES_VAL} amostras × {len(kanjis)} classes = "
          f"{CLASSIFIER_SAMPLES_VAL * len(kanjis)} crops")
    generate_split(kanjis, fonts, CLASSIFIER_SAMPLES_VAL,
                   CLASSIFIER_VAL_DIR, CLASSIFIER_SEED_VAL, "val")

    print("\n[INFO] Geração completa!")


if __name__ == "__main__":
    main()
