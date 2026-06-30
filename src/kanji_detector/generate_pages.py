"""
generate_pages.py
=================
Script local: gera o dataset sintetico completo.

Uso:
    python -m src.kanji_detector.generate_pages
"""

import os
import math
import yaml

from config import (
    DATASET_DIR,
    TRAIN_IMG_DIR, VAL_IMG_DIR,
    TRAIN_LBL_DIR, VAL_LBL_DIR,
    PAGES_AMOUNT, VAL_SPLIT,
    FONTES_URL, FONTS_DIR
)

# Garantir o download das fontes ANTES de importar o manga109
from src.helper.fonts import download_fonts
print("[INFO] Verificando fontes...")
download_fonts(FONTES_URL, FONTS_DIR)

from src.helper.manga109 import create_synthetic_manga_images


import shutil

def criar_estrutura():
    for d in [TRAIN_IMG_DIR, VAL_IMG_DIR, TRAIN_LBL_DIR, VAL_LBL_DIR]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)
    print("Estrutura de pastas limpa e criada.")


def gerar_dataset_yaml():
    """Gera o dataset.yaml com caminhos locais. No Kaggle, train.py sobrescreve."""
    yaml_path = os.path.join(DATASET_DIR, "dataset.yaml")
    config = {
        "path": DATASET_DIR,
        "train": os.path.join("images", "train"),
        "val":   os.path.join("images", "val"),
        "nc":    1,
        "names": {0: "kanji"},
    }
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
    print(f"dataset.yaml gerado em: {yaml_path}")
    return yaml_path


def gerar_dataset():
    n_val   = max(1, math.floor(PAGES_AMOUNT * VAL_SPLIT))
    n_train = PAGES_AMOUNT - n_val
    print(f"Gerando dataset: {n_train} treino + {n_val} val = {PAGES_AMOUNT} paginas")
    print("-" * 60)

    print(f"[train] Gerando {n_train} paginas...")
    geradas_train = create_synthetic_manga_images(
        img_dir=TRAIN_IMG_DIR, lbl_dir=TRAIN_LBL_DIR,
        amount=n_train, start_idx=0,
    )
    print(f"[train] {geradas_train}/{n_train} paginas geradas")

    print(f"[val]   Gerando {n_val} paginas...")
    geradas_val = create_synthetic_manga_images(
        img_dir=VAL_IMG_DIR, lbl_dir=VAL_LBL_DIR,
        amount=n_val, start_idx=n_train,
    )
    print(f"[val]   {geradas_val}/{n_val} paginas geradas")

    print("-" * 60)
    print(f"Total gerado: {geradas_train + geradas_val} paginas")
    return geradas_train, geradas_val


def gerar_amostras_com_boxes(num_amostras=5):
    import random
    from PIL import Image, ImageDraw
    
    # Pegar imagens geradas de treino
    imagens = [f for f in os.listdir(TRAIN_IMG_DIR) if f.endswith('.png')]
    if not imagens:
        return
        
    num_amostras = min(num_amostras, len(imagens))
    selecionadas = random.sample(imagens, num_amostras)
    
    # Criar pasta de amostras
    samples_dir = os.path.join(DATASET_DIR, "samples")
    if os.path.exists(samples_dir):
        shutil.rmtree(samples_dir)
    os.makedirs(samples_dir, exist_ok=True)
    
    print(f"Gerando {num_amostras} amostras com bboxes desenhadas em: {samples_dir}")
    for idx, img_name in enumerate(selecionadas):
        img_path = os.path.join(TRAIN_IMG_DIR, img_name)
        lbl_name = img_name.replace(".png", ".txt")
        lbl_path = os.path.join(TRAIN_LBL_DIR, lbl_name)
        output_path = os.path.join(samples_dir, f"amostra_{idx+1}.png")
        
        img = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        w_img, h_img = img.size
        
        if os.path.exists(lbl_path):
            with open(lbl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        _, cx, cy, w, h = map(float, parts)
                        x_center = cx * w_img
                        y_center = cy * h_img
                        width = w * w_img
                        height = h * h_img
                        
                        x0 = x_center - width / 2
                        y0 = y_center - height / 2
                        x1 = x_center + width / 2
                        y1 = y_center + height / 2
                        
                        draw.rectangle([x0, y0, x1, y1], outline="red", width=2)
                        
        img.save(output_path)


if __name__ == "__main__":
    criar_estrutura()
    gerar_dataset()
    gerar_dataset_yaml()
    gerar_amostras_com_boxes()
    print("Dataset pronto para upload no Kaggle!")
    print(f"Pasta: {DATASET_DIR}")
