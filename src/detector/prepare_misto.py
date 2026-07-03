"""
prepare_misto.py
================
Combina o dataset Manga109 com dados sintéticos no modo 'misto'.

Uso:
    python -m src.detector.prepare_misto
"""

import os
import shutil
import yaml

from config import (
    DATA_DIR,
    MANGA109_YOLO_DIR,
    MISTO_RATIO_SINTETICO,
    FONTES_URL,
    FONTS_DIR
)

# Garantir fontes baixadas
from src.helper.fonts import download_fonts
print("[INFO] Verificando fontes...")
download_fonts(FONTES_URL, FONTS_DIR)

from src.helper.manga109 import create_synthetic_manga_images

def preparar_misto():
    misto_yolo_dir = os.path.join(DATA_DIR, "misto_yolo")
    
    # 1. Verificar se o dataset manga109_yolo existe
    if not os.path.exists(MANGA109_YOLO_DIR):
        raise FileNotFoundError(
            f"Dataset Manga109 não encontrado em '{MANGA109_YOLO_DIR}'. "
            "Por favor, execute o script de preparação do Manga109 primeiro:\n"
            "python -m src.detector.prepare_manga109"
        )
        
    print(f"[INFO] Copiando dataset Manga109 de '{MANGA109_YOLO_DIR}' para '{misto_yolo_dir}'...")
    
    # Limpar diretório destino se já existe
    if os.path.exists(misto_yolo_dir):
        shutil.rmtree(misto_yolo_dir)
    os.makedirs(misto_yolo_dir, exist_ok=True)
    
    # Copiar imagens e labels do manga109_yolo
    shutil.copytree(os.path.join(MANGA109_YOLO_DIR, "images"), os.path.join(misto_yolo_dir, "images"))
    shutil.copytree(os.path.join(MANGA109_YOLO_DIR, "labels"), os.path.join(misto_yolo_dir, "labels"))
    
    # 2. Calcular quantidade de imagens sintéticas a gerar
    train_images_dir = os.path.join(misto_yolo_dir, "images", "train")
    n_manga109_treino = len([
        f for f in os.listdir(train_images_dir) 
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])
    
    n_sintetico = int(n_manga109_treino * MISTO_RATIO_SINTETICO)
    print(f"[INFO] Imagens de treino Manga109 encontradas: {n_manga109_treino}")
    print(f"[INFO] Proporção sintética: {MISTO_RATIO_SINTETICO} (gerando {n_sintetico} imagens sintéticas)")
    
    # 3. Gerar imagens sintéticas adicionais no split de treino
    train_labels_dir = os.path.join(misto_yolo_dir, "labels", "train")
    print(f"[INFO] Gerando {n_sintetico} páginas sintéticas em {train_images_dir}...")
    
    geradas = create_synthetic_manga_images(
        img_dir=train_images_dir,
        lbl_dir=train_labels_dir,
        amount=n_sintetico,
        start_idx=1000000  # Começa com índice alto para evitar conflito de nomes
    )
    
    print(f"[INFO] {geradas}/{n_sintetico} páginas sintéticas geradas com sucesso.")
    
    # 4. Gerar data.yaml apontando para misto_yolo/
    yaml_path = os.path.join(misto_yolo_dir, "data.yaml")
    config = {
        "path": os.path.abspath(misto_yolo_dir),
        "train": "images/train",
        "val":   "images/val",
        "nc":    1,
        "names": ["text"]
    }
    
    with open(yaml_path, "w", encoding="utf-8") as f_yaml:
        yaml.dump(config, f_yaml, allow_unicode=True, default_flow_style=False)
        
    print("-" * 60)
    print("Dataset Misto pronto!")
    print(f"data.yaml gerado em: {yaml_path}")
    print(f"Diretório de saída: {misto_yolo_dir}")

if __name__ == "__main__":
    preparar_misto()
