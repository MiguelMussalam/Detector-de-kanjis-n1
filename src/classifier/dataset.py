"""
dataset.py
==========
Torch Dataset para carregar os crops sintéticos do classificador de kanji.

Consome a estrutura gerada por generate_crops.py:
    data/classifier/train/U+XXXX/NNN.png
    data/classifier/val/U+XXXX/NNN.png

Usa torchvision.datasets.ImageFolder como base — o mapeamento pasta -> classe
é feito automaticamente por ordem alfabética dos codepoints.

Retorna:
    - imagem: tensor (3, 64, 64) normalizado com stats ImageNet
    - label:  int (0 a N-1, onde N = número de classes)
"""

from typing import Callable, Optional

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from config import (
    CLASSIFIER_TRAIN_DIR, CLASSIFIER_VAL_DIR,
    CLASSIFIER_INPUT_SIZE,
    CLF_NORM_MEAN, CLF_NORM_STD,
    CLF_BATCH_SIZE, CLF_NUM_WORKERS,
)


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def build_transform() -> Callable:
    """
    Transform padrão para os crops sintéticos.

    Pipeline:
      1. Converte PIL Image (grayscale ou RGB) para tensor [0, 1]
      2. Se necessário, replica canal para 3 canais (backbone pré-treinado espera RGB)
      3. Redimensiona para (CLASSIFIER_INPUT_SIZE, CLASSIFIER_INPUT_SIZE) por garantia
      4. Normaliza com mean/std ImageNet

    Observação: sem augmentation em runtime — os crops em disco já têm degradações
    aplicadas na geração (translate, blur, ruído, contraste, JPEG, etc).
    """
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),  # replica 1 canal em 3 (idempotente para RGB)
        transforms.Resize((CLASSIFIER_INPUT_SIZE, CLASSIFIER_INPUT_SIZE)),
        transforms.ToTensor(),                        # [0, 255] uint8 -> [0, 1] float
        transforms.Normalize(mean=CLF_NORM_MEAN, std=CLF_NORM_STD),
    ])


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

def build_datasets():
    """
    Retorna (train_ds, val_ds) usando ImageFolder.

    ImageFolder ordena as classes alfabeticamente por nome de pasta. Como usamos
    'U+XXXX' como nome, a ordenação é determinística e estável entre execuções.
    """
    transform = build_transform()

    train_ds = ImageFolder(root=CLASSIFIER_TRAIN_DIR, transform=transform)
    val_ds   = ImageFolder(root=CLASSIFIER_VAL_DIR,   transform=transform)

    # Sanity check: mesmas classes nos dois splits
    assert train_ds.classes == val_ds.classes, (
        f"Classes divergem entre train e val:\n"
        f"  train tem {len(train_ds.classes)} classes\n"
        f"  val   tem {len(val_ds.classes)} classes"
    )

    return train_ds, val_ds


# ---------------------------------------------------------------------------
# DataLoaders
# ---------------------------------------------------------------------------

def build_dataloaders(train_ds: ImageFolder, val_ds: ImageFolder,
                      batch_size: Optional[int] = None,
                      num_workers: Optional[int] = None):
    """
    Retorna (train_loader, val_loader) prontos pra usar no loop de treino.
    """
    batch_size  = batch_size  or CLF_BATCH_SIZE
    num_workers = num_workers or CLF_NUM_WORKERS

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Helper: mapear índice de classe <-> codepoint <-> kanji
# ---------------------------------------------------------------------------

def index_to_kanji(class_names: list, index: int) -> str:
    """
    Recebe class_names de ImageFolder (lista de nomes de pasta 'U+XXXX')
    e um índice de classe, retorna o caractere kanji.
    """
    codepoint_str = class_names[index]  # 'U+5B66'
    codepoint = int(codepoint_str.replace("U+", ""), 16)
    return chr(codepoint)


def kanji_to_index(class_to_idx: dict, kanji: str) -> int:
    """
    Recebe class_to_idx de ImageFolder e um caractere, retorna o índice.
    """
    codepoint_str = f"U+{ord(kanji):04X}"
    return class_to_idx[codepoint_str]


# ---------------------------------------------------------------------------
# CLI: sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Carregando datasets...")
    train_ds, val_ds = build_datasets()

    print(f"Treino: {len(train_ds)} amostras, {len(train_ds.classes)} classes")
    print(f"Val:    {len(val_ds)} amostras, {len(val_ds.classes)} classes")
    print(f"Primeiras 5 classes: {train_ds.classes[:5]}")
    print(f"Primeiro kanji: {index_to_kanji(train_ds.classes, 0)} (classe 0)")

    # Testa carregar uma amostra
    img, label = train_ds[0]
    print(f"\nAmostra 0:")
    print(f"  Shape:  {img.shape}")
    print(f"  Dtype:  {img.dtype}")
    print(f"  Range:  [{img.min():.3f}, {img.max():.3f}]")
    print(f"  Label:  {label} ({train_ds.classes[label]} = {index_to_kanji(train_ds.classes, label)})")

    # Testa dataloader
    print("\nCriando dataloaders...")
    train_loader, val_loader = build_dataloaders(train_ds, val_ds)
    print(f"Batches treino: {len(train_loader)}")
    print(f"Batches val:    {len(val_loader)}")

    batch_imgs, batch_labels = next(iter(train_loader))
    print(f"\nPrimeiro batch de treino:")
    print(f"  Imgs shape:   {batch_imgs.shape}")
    print(f"  Labels shape: {batch_labels.shape}")