"""
generate_pages.py
=================
Script local: gera o dataset sintetico completo.

Uso:
    python -m src.detector.generate_pages
"""

import os
import yaml

from config import (
    DATASET_DIR,
    MANGA109_YOLO_DIR,
    TRAIN_IMG_DIR, TRAIN_LBL_DIR,
    PAGES_AMOUNT,
    FONTES_URL, FONTS_DIR
)

# Garantir o download das fontes ANTES de importar o manga109
from src.helper.fonts import download_fonts
print("[INFO] Verificando fontes...")
download_fonts(FONTES_URL, FONTS_DIR)

from src.helper.manga109 import create_synthetic_manga_images


import shutil

def criar_estrutura():
    # Apenas treino — val é sempre os 10 volumes reais do Manga109
    for d in [TRAIN_IMG_DIR, TRAIN_LBL_DIR]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)
    print("Estrutura de pastas limpa e criada.")


def gerar_dataset_yaml():
    """Gera o data.yaml. val aponta para os 10 volumes reais do Manga109."""
    val_real = os.path.abspath(os.path.join(MANGA109_YOLO_DIR, "images", "val"))
    yaml_path = os.path.join(DATASET_DIR, "data.yaml")
    config = {
        "path": os.path.abspath(DATASET_DIR),
        "train": "images/train",
        "val":   val_real,
        "nc":    1,
        "names": ["text"],
    }
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
    print(f"data.yaml gerado em: {yaml_path}")
    print(f"  train: {os.path.abspath(DATASET_DIR)}/images/train")
    print(f"  val:   {val_real}")
    return yaml_path


def gerar_dataset():
    print(f"Gerando dataset: {PAGES_AMOUNT} paginas de treino")
    print("-" * 60)

    print(f"[train] Gerando {PAGES_AMOUNT} paginas...")
    geradas_train = create_synthetic_manga_images(
        img_dir=TRAIN_IMG_DIR, lbl_dir=TRAIN_LBL_DIR,
        amount=PAGES_AMOUNT, start_idx=0,
    )
    print(f"[train] {geradas_train}/{PAGES_AMOUNT} paginas geradas")

    print("-" * 60)
    print(f"Total gerado: {geradas_train} paginas")
    return geradas_train, 0


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
