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
)
from src.helper.manga109 import create_synthetic_manga_images


def criar_estrutura():
    for d in [TRAIN_IMG_DIR, VAL_IMG_DIR, TRAIN_LBL_DIR, VAL_LBL_DIR]:
        os.makedirs(d, exist_ok=True)
    print("Estrutura de pastas criada.")


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


if __name__ == "__main__":
    criar_estrutura()
    gerar_dataset()
    gerar_dataset_yaml()
    print("Dataset pronto para upload no Kaggle!")
    print(f"Pasta: {DATASET_DIR}")
