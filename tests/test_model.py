"""
Testes de src/classifier/model.py -- foco na flag stem_leve (adaptacao do
stem do ResNet-18 pra entrada pequena, ver CLF_STEM_LEVE em config.py).

O stem padrao do ResNet foi desenhado pra 224x224 (ImageNet); com a nossa
entrada de 64x64, o mesmo downsample agressivo comprime a imagem num mapa de
2x2 antes do avgpool -- perde justamente o detalhe fino de traco que
diferencia kanji complexos. stem_leve muda so o stride do conv1 (mantendo o
kernel/pesos pre-treinados intactos) e troca o maxpool por Identity.
"""

import torch

from src.classifier.model import build_model, count_parameters


def _mapa_antes_do_avgpool(model, dummy):
    mapa = {}

    def hook(module, inp, out):
        mapa["shape"] = tuple(out.shape[-2:])

    handle = model.layer4.register_forward_hook(hook)
    with torch.no_grad():
        model(dummy)
    handle.remove()
    return mapa["shape"]


def test_stem_padrao_comprime_para_2x2():
    """Comportamento de hoje (default) -- documenta o problema que stem_leve resolve."""
    model = build_model(num_classes=10, stem_leve=False)
    dummy = torch.randn(2, 3, 64, 64)
    assert _mapa_antes_do_avgpool(model, dummy) == (2, 2)


def test_stem_leve_preserva_8x8():
    """Com stem_leve, o mapa final fica 16x maior (8x8 em vez de 2x2)."""
    model = build_model(num_classes=10, stem_leve=True)
    dummy = torch.randn(2, 3, 64, 64)
    assert _mapa_antes_do_avgpool(model, dummy) == (8, 8)


def test_stem_leve_nao_muda_quantidade_de_parametros():
    """So muda stride (config) e substitui maxpool por Identity (sem peso) -- zero parametro a mais/menos."""
    modelo_padrao = build_model(num_classes=10, stem_leve=False)
    modelo_leve = build_model(num_classes=10, stem_leve=True)
    assert count_parameters(modelo_padrao) == count_parameters(modelo_leve)


def test_stem_leve_produz_saida_com_shape_correto():
    num_classes = 37
    model = build_model(num_classes=num_classes, stem_leve=True)
    dummy = torch.randn(4, 3, 64, 64)
    with torch.no_grad():
        out = model(dummy)
    assert out.shape == (4, num_classes)
