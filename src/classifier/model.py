"""
model.py
========
Modelo de classificação de kanji baseado em backbone CNN pré-treinado.

Padrão: ResNet-18 do torchvision, pré-treinado em ImageNet, com a última camada
fully-connected substituída para o número de classes do dataset (1232 kanji N1
por padrão).

Uso:
    from src.classifier.model import build_model
    model = build_model(num_classes=1232)
"""

import torch
import torch.nn as nn
from torchvision import models

from config import CLF_MODEL_ARCH, CLF_PRETRAINED


# ---------------------------------------------------------------------------
# Backbones suportados
# ---------------------------------------------------------------------------

def _build_resnet18(num_classes: int, pretrained: bool) -> nn.Module:
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)
    # Substitui a cabeça (fc) para o número de classes desejado
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def _build_efficientnet_b0(num_classes: int, pretrained: bool) -> nn.Module:
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    # A cabeça do EfficientNet é um Sequential; o Linear é o último elemento
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


# ---------------------------------------------------------------------------
# Fábrica
# ---------------------------------------------------------------------------

BACKBONES = {
    "resnet18":         _build_resnet18,
    "efficientnet_b0":  _build_efficientnet_b0,
}


def build_model(num_classes: int,
                arch: str = None,
                pretrained: bool = None) -> nn.Module:
    """
    Constrói o modelo do classificador.

    Args:
        num_classes: Número de classes (kanji) na cabeça de classificação.
        arch:        Arquitetura do backbone. Se None, usa CLF_MODEL_ARCH do config.
        pretrained:  Se True, carrega pesos pré-treinados ImageNet.
                     Se None, usa CLF_PRETRAINED do config.

    Returns:
        nn.Module pronto pra treinar.
    """
    arch       = arch       or CLF_MODEL_ARCH
    pretrained = CLF_PRETRAINED if pretrained is None else pretrained

    if arch not in BACKBONES:
        raise ValueError(
            f"Arquitetura '{arch}' não suportada. "
            f"Opções: {list(BACKBONES.keys())}"
        )

    builder = BACKBONES[arch]
    return builder(num_classes=num_classes, pretrained=pretrained)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def count_parameters(model: nn.Module) -> int:
    """Retorna o número total de parâmetros treináveis do modelo."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# CLI: sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    NUM_CLASSES = 1232  # kanji N1

    print(f"Construindo modelo: arch={CLF_MODEL_ARCH}, pretrained={CLF_PRETRAINED}")
    model = build_model(num_classes=NUM_CLASSES)

    n_params = count_parameters(model)
    print(f"Parâmetros treináveis: {n_params:,} ({n_params/1e6:.2f}M)")

    # Testa forward pass com input dummy
    print("\nTestando forward pass...")
    dummy = torch.randn(2, 3, 64, 64)  # batch=2, 3 canais, 64x64
    with torch.no_grad():
        out = model(dummy)
    print(f"Input shape:  {dummy.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Output range: [{out.min():.3f}, {out.max():.3f}]")

    assert out.shape == (2, NUM_CLASSES), \
        f"Shape esperado (2, {NUM_CLASSES}), got {out.shape}"

    print("\n[OK] Modelo funcional.")