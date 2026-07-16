"""
synth_page.py
==============
Compõe glifos sintéticos em cima de páginas REAIS do Manga109, ancorados nas
bboxes de linha oficiais (<text>) -- resolve o problema estrutural do gerador
antigo (deletado no commit 8292b18), que colava blocos de texto em posições
arbitrárias da página sem checar contra a anotação oficial, arriscando deixar
texto real não rotulado em outro canto do mesmo recorte.

Para cada <text> de uma página:
  1. Subdivide a bbox em N células iguais (N = número de caracteres) -- só
     quando a geometria (tamanho de célula + razão de largura) é plausível
     pra coluna/linha única (ver DETSYN_CELL_*/DETSYN_MAX_RAZAO_LARGURA em
     config.py). Manga109 não diz onde cada coluna começa dentro de uma
     bbox multi-coluna, então essas linhas são só apagadas, sem rótulo --
     nunca se tenta adivinhar o layout.
  2. Apaga a tinta original de cada célula usada (cv2.inpaint, cor de tinta
     amostrada da própria região) e desenha um glifo sintético novo,
     reaproveitando as primitivas de render/degradação já maduras de
     src/classifier/generate_crops.py.
  3. Bbox de saída (formato YOLO, classe única "glifo") calculada por
     threshold no glifo já renderizado/transformado, não por transformação
     da bbox original -- não precisa de biblioteca bbox-aware.

Cada linha usada fica 100% rotulada (todas as células têm caixa); linhas
puladas por geometria só têm a tinta apagada, sem rótulo -- nunca sobra
tinta real sem caixa, que ensinaria o detector a ignorar caracteres de
verdade.
"""

import random

import cv2
import numpy as np
from PIL import Image

from config import (
    DETSYN_CELL_MIN_PX, DETSYN_CELL_MAX_PX, DETSYN_MAX_RAZAO_LARGURA,
    DETSYN_APAGAR_MARGEM_PX, DETSYN_PROP_ORIGINAL_CHAR,
    CLF_MIN_FILL_RATIO, CLF_MAX_FILL_RATIO,
)
from src.helper.manga109_align import limpar_string
from src.classifier.generate_crops import render_glyph, apply_morphology, apply_rotation, _fill_ratio_glifo


def bbox_para_yolo(bbox, img_w, img_h):
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2 / img_w
    cy = (y1 + y2) / 2 / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return cx, cy, w, h


def dividir_em_celulas(bbox, n_chars: int):
    """
    Divide a bbox em n_chars células iguais ao longo do eixo de leitura.
    Retorna lista de bbox (uma por célula) ou None se a geometria não for
    plausível pra coluna/linha única.
    """
    x1, y1, x2, y2 = bbox
    largura, altura = x2 - x1, y2 - y1
    vertical = altura >= largura

    lado_celula = (altura if vertical else largura) / n_chars
    outro_lado = largura if vertical else altura

    if not (DETSYN_CELL_MIN_PX <= lado_celula <= DETSYN_CELL_MAX_PX):
        return None
    if outro_lado > lado_celula * DETSYN_MAX_RAZAO_LARGURA:
        return None  # provavel multi-coluna -- Manga109 nao diz onde cada uma comeca

    celulas = []
    for i in range(n_chars):
        if vertical:
            celulas.append((x1, y1 + i * lado_celula, x2, y1 + (i + 1) * lado_celula))
        else:
            celulas.append((x1 + i * lado_celula, y1, x1 + (i + 1) * lado_celula, y2))
    return celulas


def _mascara_e_cor_tinta(regiao_gray: np.ndarray):
    """
    Separa tinta de fundo via Otsu e assume que a tinta é o cluster
    MINORITÁRIO de pixels (a maior parte de uma bbox de fala é fundo/balão,
    não traço) -- funciona pra polaridade normal (traço escuro) e invertida
    (traço claro sobre painel escuro) sem caso especial.
    """
    if regiao_gray.size == 0:
        return np.zeros_like(regiao_gray, dtype=bool), 0
    _, mascara = cv2.threshold(regiao_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mascara_bin = mascara > 0
    tinta = mascara_bin if mascara_bin.mean() < 0.5 else ~mascara_bin
    cor_tinta = int(regiao_gray[tinta].mean()) if tinta.any() else int(regiao_gray.mean())
    return tinta, cor_tinta


def apagar_linha(frame_gray: np.ndarray, bbox):
    """
    Apaga (inpaint) a tinta original dentro da bbox (+margem). Retorna
    (frame_atualizado, cor_tinta_amostrada) -- cor usada pro glifo novo.

    Roda pra TODA linha <text>, mesmo quando dividir_em_celulas() depois
    rejeita a geometria (multi-coluna) e nenhum glifo novo e' desenhado ali --
    e' isso que garante a invariante "nunca sobra tinta real legivel sem
    caixa". Em blocos densos multi-linha o inpaint nao produz papel limpo,
    vira uma textura pontilhada/borrada (auditado visualmente) -- pior
    esteticamente que um fundo liso, mas ainda assim ilegivel como texto, o
    que preserva a invariante (nao ensina o detector a ignorar caractere real
    formado).
    """
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    m = DETSYN_APAGAR_MARGEM_PX
    x1e, y1e = max(0, x1 - m), max(0, y1 - m)
    x2e, y2e = min(frame_gray.shape[1], x2 + m), min(frame_gray.shape[0], y2 + m)

    regiao = frame_gray[y1e:y2e, x1e:x2e]
    tinta, cor_tinta = _mascara_e_cor_tinta(regiao)
    if not tinta.any():
        return frame_gray, cor_tinta

    reparado = cv2.inpaint(regiao, (tinta.astype(np.uint8)) * 255, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    frame_gray = frame_gray.copy()
    frame_gray[y1e:y2e, x1e:x2e] = reparado
    return frame_gray, cor_tinta


def _renderizar_tentativa(char: str, font_path: str, cell_w: int, cell_h: int, rng: random.Random):
    font_size = max(8, min(cell_w, cell_h))
    canvas_size = max(cell_w, cell_h, font_size)

    img = render_glyph(char, font_path, font_size, canvas_size)
    img = apply_morphology(img, rng, font_path=font_path)
    img = apply_rotation(img, rng)
    return np.array(Image.fromarray(img).resize((cell_w, cell_h), resample=Image.BILINEAR))


def renderizar_glifo_para_celula(char: str, font_path: str, cor_tinta: int,
                                  rng: random.Random, cell_w: int, cell_h: int,
                                  fonts_fallback: list = None):
    """
    Renderiza um glifo (reaproveita render_glyph/apply_morphology/
    apply_rotation de generate_crops.py) do tamanho da célula, com a cor de
    tinta amostrada da linha original. Retorna (patch, bbox_tight_relativa)
    ou (None, None) se nao conseguir um render legivel.

    Checa fill ratio (fração de "tinta" dentro da própria bbox do glifo, não
    contagem absoluta de pixel -- células aqui variam muito de tamanho, ao
    contrário dos crops 64x64 fixos do classificador) em vez de só "mask não
    vazia": pega o caso de fonte sem esse caractere (render vira retângulo
    "tofu" sólido, comum quando o caractere original da transcrição -- que
    pode ser hiragana/katakana/kanji fora do N1 -- não é coberto pela fonte
    sorteada, já que só a cobertura N1 foi verificada em verify_fonts_compatibility).
    """
    tentativas = [font_path] + list(fonts_fallback or [])[:3]
    for fp in tentativas:
        img = _renderizar_tentativa(char, fp, cell_w, cell_h, rng)
        mask = img <= 240
        if not mask.any():
            continue
        if not (CLF_MIN_FILL_RATIO <= _fill_ratio_glifo(img, mask) <= CLF_MAX_FILL_RATIO):
            continue

        linhas = np.where(mask.any(axis=1))[0]
        colunas = np.where(mask.any(axis=0))[0]
        bbox_rel = (int(colunas[0]), int(linhas[0]), int(colunas[-1] + 1), int(linhas[-1] + 1))

        patch = np.full((cell_h, cell_w), 255, dtype=np.uint8)
        patch[mask] = cor_tinta
        return patch, bbox_rel

    return None, None


def processar_pagina_para_synth(frame_bgr: np.ndarray, text_elements: list,
                                 n1_kanjis: list, fonts: list, rng: random.Random):
    """
    Processa uma página inteira: para cada <text>, tenta subdividir em
    células e substituir por glifos sintéticos; se a geometria não permitir
    (linha multi-coluna), só apaga a tinta, sem rótulo. Retorna
    (frame_resultado_bgr, lista_de_bboxes_absolutas_x1y1x2y2).
    """
    frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    bboxes_saida = []

    for text_el in text_elements:
        texto_limpo = limpar_string(text_el.text)
        if not texto_limpo:
            continue

        bbox = (
            int(text_el.get("xmin")), int(text_el.get("ymin")),
            int(text_el.get("xmax")), int(text_el.get("ymax")),
        )

        frame_gray, cor_tinta = apagar_linha(frame_gray, bbox)

        celulas = dividir_em_celulas(bbox, len(texto_limpo))
        if celulas is None:
            continue  # geometria nao confiavel -- so apaga, sem rotulo

        for i, (cx1, cy1, cx2, cy2) in enumerate(celulas):
            usar_original = rng.random() < DETSYN_PROP_ORIGINAL_CHAR
            char = texto_limpo[i] if usar_original else rng.choice(n1_kanjis)
            font_path = rng.choice(fonts)

            cell_w = max(1, int(round(cx2 - cx1)))
            cell_h = max(1, int(round(cy2 - cy1)))

            fallback = [f for f in fonts if f != font_path]
            rng.shuffle(fallback)
            patch, bbox_rel = renderizar_glifo_para_celula(
                char, font_path, cor_tinta, rng, cell_w, cell_h, fonts_fallback=fallback
            )
            if patch is None:
                continue  # nenhuma fonte tentada rendeu um glifo legivel -- pula a celula

            x0, y0 = int(round(cx1)), int(round(cy1))
            frame_gray[y0:y0 + cell_h, x0:x0 + cell_w] = patch

            bx1, by1, bx2, by2 = bbox_rel
            bboxes_saida.append((x0 + bx1, y0 + by1, x0 + bx2, y0 + by2))

    frame_resultado = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
    return frame_resultado, bboxes_saida
