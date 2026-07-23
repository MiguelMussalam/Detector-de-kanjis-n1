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

from config import CLF_MODEL_ARCH, CLF_PRETRAINED, CLF_STEM_LEVE

def _build_resnet18(num_classes: int, pretrained: bool, stem_leve: bool = False) -> nn.Module:
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)

    if stem_leve:
        # So muda o stride do conv1 (mesmo kernel/pesos pre-treinados, so a
        # config de forward muda) e troca o maxpool por Identity -- ver
        # CLF_STEM_LEVE em config.py pro raciocinio completo. Mapa antes do
        # avgpool passa de 2x2 pra 8x8 (entrada 64x64).
        model.conv1.stride = (1, 1)
        model.maxpool = nn.Identity()

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def _build_efficientnet_b0(num_classes: int, pretrained: bool, stem_leve: bool = False) -> nn.Module:
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    # A cabeça do EfficientNet é um Sequential; o Linear é o último elemento
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model

BACKBONES = {
    "resnet18":         _build_resnet18,
    "efficientnet_b0":  _build_efficientnet_b0,
}


def build_model(num_classes: int,
                arch: str = None,
                pretrained: bool = None,
                stem_leve: bool = None) -> nn.Module:
    """
    Constrói o modelo do classificador.

    Args:
        num_classes: Número de classes (kanji) na cabeça de classificação.
        arch:        Arquitetura do backbone. Se None, usa CLF_MODEL_ARCH do config.
        pretrained:  Se True, carrega pesos pré-treinados ImageNet.
                     Se None, usa CLF_PRETRAINED do config.
        stem_leve:   Se True (só resnet18), stride do conv1 vira 1 e o maxpool
                     inicial vira Identity -- ver CLF_STEM_LEVE em config.py.
                     Se None, usa CLF_STEM_LEVE do config. Ignorado por outras
                     arquiteturas (ex: efficientnet_b0).

    Returns:
        nn.Module pronto pra treinar.
    """
    arch       = arch       or CLF_MODEL_ARCH
    pretrained = CLF_PRETRAINED if pretrained is None else pretrained
    stem_leve  = CLF_STEM_LEVE if stem_leve is None else stem_leve

    if arch not in BACKBONES:
        raise ValueError(
            f"Arquitetura '{arch}' não suportada. "
            f"Opções: {list(BACKBONES.keys())}"
        )

    builder = BACKBONES[arch]
    return builder(num_classes=num_classes, pretrained=pretrained, stem_leve=stem_leve)

def count_parameters(model: nn.Module) -> int:
    """Retorna o número total de parâmetros treináveis do modelo."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    NUM_CLASSES = 1232  # kanji N1

    for stem_leve in (False, True):
        print(f"\nConstruindo modelo: arch={CLF_MODEL_ARCH}, pretrained={CLF_PRETRAINED}, "
              f"stem_leve={stem_leve}")
        model = build_model(num_classes=NUM_CLASSES, stem_leve=stem_leve)

        n_params = count_parameters(model)
        print(f"Parâmetros treináveis: {n_params:,} ({n_params/1e6:.2f}M)")

        dummy = torch.randn(2, 3, 64, 64)  # batch=2, 3 canais, 64x64

        mapa_final = {}
        def _hook(module, inp, out):
            mapa_final["shape"] = tuple(out.shape[-2:])
        handle = model.layer4.register_forward_hook(_hook)

        with torch.no_grad():
            out = model(dummy)
        handle.remove()

        print(f"Input shape:  {dummy.shape}")
        print(f"Mapa antes do avgpool (layer4 output): {mapa_final['shape']}")
        print(f"Output shape: {out.shape}")
        print(f"Output range: [{out.min():.3f}, {out.max():.3f}]")

        assert out.shape == (2, NUM_CLASSES), \
            f"Shape esperado (2, {NUM_CLASSES}), got {out.shape}"

    print("\n[OK] Modelo funcional (com e sem stem_leve).")