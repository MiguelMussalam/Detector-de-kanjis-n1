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

# Console do Windows (cp1252) nao imprime kanji direto -- os prints de [WARN]
# abaixo tem caractere japones no meio da mensagem, sem isso o processo
# inteiro morre com UnicodeEncodeError no primeiro aviso.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from config import (
    CLASSIFIER_TRAIN_DIR, CLASSIFIER_VAL_DIR, CLASSIFIER_SANITY_DIR,
    CLASSIFIER_KANJI_LEVEL, CLASSIFIER_INPUT_SIZE,
    CLASSIFIER_SAMPLES_TRAIN, CLASSIFIER_SAMPLES_VAL, CLASSIFIER_SANITY_COUNT,
    CLASSIFIER_SEED_TRAIN, CLASSIFIER_SEED_VAL,
    CLASSIFIER_FONT_SIZES, CLASSIFIER_FONT_SIZE_WEIGHTS, CLASSIFIER_CANVAS_MARGIN,
    CLF_BG_VALUE_MIN, CLF_BG_VALUE_MAX, CLF_BG_NOISE_STD,
    BACKGROUNDS_REAL_DIR, CLF_BG_ESCURO_PROB,
    CLF_TRANSLATE_PROB, CLF_TRANSLATE_MAX,
    CLF_BLUR_PROB, CLF_BLUR_SIGMA_MIN, CLF_BLUR_SIGMA_MAX,
    CLF_NOISE_PROB, CLF_NOISE_STD_MIN, CLF_NOISE_STD_MAX,
    CLF_BRIGHTNESS_PROB, CLF_BRIGHTNESS_RANGE,
    CLF_CONTRAST_PROB, CLF_CONTRAST_RANGE,
    CLF_MORFO_PROB, CLF_MORFO_K_MIN, CLF_MORFO_K_MAX,
    CLF_JPEG_PROB, CLF_JPEG_QUALITY_MIN, CLF_JPEG_QUALITY_MAX,
    CLF_ROTATION_PROB, CLF_ROTATION_MAX,
    CLF_OUTROS_LABEL, CLF_OUTROS_SAMPLES_TRAIN, CLF_OUTROS_SAMPLES_VAL,
    CLF_OUTROS_PROP_KANJI, CLF_OUTROS_PROP_KANA, CLF_OUTROS_PROP_LATIM, CLF_OUTROS_PROP_RUIDO,
    CLF_MIN_CONTRASTE, CLF_MAX_TENTATIVAS, CLF_MIN_PIXELS_GLIFO,
    CLF_MIN_FILL_RATIO, CLF_MAX_FILL_RATIO, CLF_MAX_TENTATIVAS_MORFO, CLF_FONTES_PESADAS,
)
from src.helper.kanjis import get_kanjis
from src.helper.fonts import get_fonts_list


def render_glyph(char: str, font_path: str, font_size: int,
                 canvas_size: int) -> np.ndarray:
    """
    Renderiza um caractere em canvas quadrado, glyph centralizado, fundo branco.
    Retorna array uint8 (H, W) em grayscale.
    """
    font = ImageFont.truetype(font_path, size=font_size)
    canvas = Image.new("L", (canvas_size, canvas_size), color=255)
    draw = ImageDraw.Draw(canvas)

    bbox = draw.textbbox((0, 0), char, font=font)
    glyph_w = bbox[2] - bbox[0]
    glyph_h = bbox[3] - bbox[1]

    x = (canvas_size - glyph_w) // 2 - bbox[0]
    y = (canvas_size - glyph_h) // 2 - bbox[1]

    draw.text((x, y), char, font=font, fill=0)
    return np.array(canvas, dtype=np.uint8)


def apply_morphology(img: np.ndarray, rng: random.Random, font_path: str = None) -> np.ndarray:
    """
    Dilate ou erode leve com kernel pequeno.

    Fontes bold/pesadas (CLF_FONTES_PESADAS) já são grossas por design --
    "dilate" (engrossar ainda mais) é o que faz kanji com muitos traços
    grudarem numa mancha sólida (achado na auditoria visual). Pra essas, só
    "erode" (afinar) é permitido.
    """
    if rng.random() > CLF_MORFO_PROB:
        return img
    k = rng.randint(CLF_MORFO_K_MIN, CLF_MORFO_K_MAX)
    eh_pesada = font_path is not None and os.path.basename(font_path) in CLF_FONTES_PESADAS
    op = "erode" if eh_pesada else rng.choice(["dilate", "erode"])
    # Glyph é preto (0) sobre branco (255): a operação morfológica "dilate" (que
    # expandiria o preto em imagens normais) precisa ser grey_erosion aqui, e
    # vice-versa -- os nomes das funções do scipy pressupõem primeiro-plano claro.
    if op == "dilate":
        return grey_erosion(img, size=(k, k))
    else:
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

    cx = w // 2 + dx
    cy = h // 2 + dy

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


_FUNDOS_REAIS_CACHE = {}


def _carregar_fundos_reais(subpasta: str) -> list:
    """Carrega (uma vez, com cache) os patches de fundo real de uma subpasta (claro/escuro)."""
    if subpasta not in _FUNDOS_REAIS_CACHE:
        pasta = os.path.join(BACKGROUNDS_REAL_DIR, subpasta)
        arquivos = sorted(Path(pasta).glob("*.png")) if os.path.isdir(pasta) else []
        if not arquivos:
            print(f"[AVISO] Nenhum fundo real encontrado em '{pasta}' -- caindo para o "
                  f"fundo sintetico (papel liso). Anexe o dataset de backgrounds_real "
                  f"(ver src/helper/harvest_backgrounds.py) se essa nao for a intencao.")
        _FUNDOS_REAIS_CACHE[subpasta] = [
            np.array(Image.open(f).convert("L")) for f in arquivos
        ]
    return _FUNDOS_REAIS_CACHE[subpasta]


def _recortar_fundo(bg_full: np.ndarray, h: int, w: int, rng: random.Random) -> np.ndarray:
    """Recorta um pedaço aleatorio de bg_full do tamanho (h, w) -- o patch harvested
    é maior que o input de proposito, pra dar variedade mesmo com poucos arquivos."""
    bh, bw = bg_full.shape
    if bh >= h and bw >= w:
        y0 = rng.randint(0, bh - h)
        x0 = rng.randint(0, bw - w)
        return bg_full[y0:y0 + h, x0:x0 + w]
    return np.array(Image.fromarray(bg_full).resize((w, h), resample=Image.BILINEAR))


def apply_paper_background(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """
    Aplica fundo real (recortado do Manga109, ver src/helper/harvest_backgrounds.py)
    quando o pool estiver disponível; cai para o fundo sintético (papel liso)
    como fallback (ex: ambiente sem esse dataset coletado ainda).

    Fundo ESCURO exige inverter a polaridade do glyph (traço branco sobre
    fundo preto) -- do contrário o traço preto original fica invisível sobre
    o fundo preto. Fundo CLARO usa a mesma lógica de sempre (preserva o traço
    escuro, preenche o resto com o fundo).
    """
    h, w = img.shape

    usar_escuro = rng.random() < CLF_BG_ESCURO_PROB
    fundos = _carregar_fundos_reais("escuro" if usar_escuro else "claro")

    if fundos:
        bg = _recortar_fundo(rng.choice(fundos), h, w, rng)
        if usar_escuro:
            img = 255 - img  # inverte: traço vira branco, fundo original (branco) vira preto
            mask = img < 15
        else:
            mask = img > 240
        out = img.copy()
        out[mask] = bg[mask]
        return out

    # Fallback: sem pool de fundo real coletado, usa papel liso sintético.
    bg_value = rng.uniform(CLF_BG_VALUE_MIN, CLF_BG_VALUE_MAX)
    bg = np.full((h, w), bg_value, dtype=np.float32)
    noise = np.random.normal(0, CLF_BG_NOISE_STD, (h, w))
    bg = np.clip(bg + noise, 0, 255).astype(np.uint8)
    mask = img > 240
    out = img.copy()
    out[mask] = bg[mask]
    return out


def _fg_bg_means(img: np.ndarray, mask_glifo: np.ndarray):
    """Media de brilho dentro/fora da mascara do glifo, ou None se nao ha o que comparar
    (mascara vazia ou cobrindo a imagem inteira -- glifo saiu do crop)."""
    if mask_glifo.sum() == 0 or (~mask_glifo).sum() == 0:
        return None
    return (img[mask_glifo].astype(np.float32).mean(),
            img[~mask_glifo].astype(np.float32).mean())


def _contraste_glifo(img: np.ndarray, mask_glifo: np.ndarray) -> float:
    """
    Diferença média (escala 0-255) entre a região do glifo e o resto da
    imagem -- proxy barato de legibilidade. Se cair demais, a degradação
    "esvaziou" o caractere e a amostra virou ruído com o rótulo da classe
    (pior que não ter a amostra: ensina uma associação errada).
    """
    medias = _fg_bg_means(img, mask_glifo)
    if medias is None:
        return 255.0  # nada pra comparar (glifo saiu inteiro do crop) -- deixa passar
    fg, bg = medias
    return abs(fg - bg)


def _fill_ratio_glifo(img: np.ndarray, mask_glifo: np.ndarray) -> float:
    """
    Fração de pixel "tinta" dentro do retângulo delimitador do próprio glifo
    (não do canvas inteiro) -- complementar ao _contraste_glifo, que só mede
    brilho médio e fica cego pra dois jeitos de perder legibilidade que ainda
    passam nele: traços que grudaram numa mancha sólida (fill ratio alto
    demais) ou que se desfizeram em fiapos esparsos (fill ratio baixo demais).

    Usa o ponto médio entre as médias fg/bg (calculadas na hora) como limiar,
    então funciona igual pro caso de fundo escuro/glifo invertido, sem
    precisar de caso especial pra polaridade.
    """
    linhas = np.where(mask_glifo.any(axis=1))[0]
    colunas = np.where(mask_glifo.any(axis=0))[0]
    if linhas.size == 0 or colunas.size == 0:
        return 0.0
    medias = _fg_bg_means(img, mask_glifo)
    if medias is None:
        return 0.0
    fg_media, bg_media = medias
    regiao = img[linhas[0]:linhas[-1] + 1, colunas[0]:colunas[-1] + 1].astype(np.float32)
    limiar = (fg_media + bg_media) / 2
    e_tinta = (regiao <= limiar) if fg_media <= bg_media else (regiao >= limiar)
    return float(e_tinta.mean())


def _legivel(img: np.ndarray, mask_glifo: np.ndarray) -> bool:
    """Combina as duas checagens de legibilidade -- usada tanto no render base
    quanto na degradação final (ver generate_sample)."""
    return (mask_glifo.sum() >= CLF_MIN_PIXELS_GLIFO and
            CLF_MIN_FILL_RATIO <= _fill_ratio_glifo(img, mask_glifo) <= CLF_MAX_FILL_RATIO)


def _aplicar_fundo_e_degradacoes(img_base: np.ndarray, rng: random.Random,
                                 aplicar_blur_ruido: bool = True) -> np.ndarray:
    """Passos 6-10 (fundo + degradações finais), a partir do glifo já
    renderizado/posicionado. Separado pra permitir re-sortear só essa parte
    (ver generate_sample) sem re-sortear fonte/rotação/posição a cada tentativa."""
    img = apply_paper_background(img_base.copy(), rng)
    if aplicar_blur_ruido:
        img = apply_blur(img, rng)
        img = apply_noise(img, rng)
    img = apply_brightness_contrast(img, rng)
    img = apply_jpeg(img, rng)
    return img


def _renderizar_base(char: str, font_path: str, font_size: int,
                     canvas_margin: float, rng: random.Random, output_size: int) -> np.ndarray:
    """Render + degradações geométricas, antes do fundo/blur (separado do resto
    do pipeline pra permitir retry só na parte final, ver generate_sample)."""
    canvas_size = int(font_size * (1 + 2 * canvas_margin))
    img = render_glyph(char, font_path, font_size, canvas_size)
    img = apply_morphology(img, rng, font_path=font_path)
    img = apply_rotation(img, rng)
    img = apply_translate_and_crop(img, rng, output_size)
    return resize_to_target(img, output_size)


def generate_sample(char: str, font_path: str, font_size: int,
                    rng: random.Random, output_size: int,
                    canvas_margin: float, fonts_fallback: list = None,
                    avisos: list = None) -> np.ndarray:
    """
    Gera um crop 64x64 do kanji com degradações aplicadas em ordem.

    Garantias contra amostra ilegível (sobra só textura, ou virou uma mancha
    sólida/fiapos esparsos sem estrutura de traço reconhecível):
      1. Antes de degradar: algumas fontes rendem em branco pra um caractere
         raro (glifo ausente/quebrado naquele tamanho), ou a combinação de
         fonte pesada + morfologia + tamanho pequeno gruda os traços numa
         mancha (ou o oposto, esfarela em fiapos). Re-sorteia geometria
         (mesma fonte, `CLF_MAX_TENTATIVAS_MORFO` vezes) e, se não resolver,
         tenta outras fontes de `fonts_fallback` -- essa etapa roda de novo
         porque a fonte/morfologia só é sorteada uma vez, então nenhuma
         tentativa da etapa 2 abaixo consegue consertar um render que já
         nasceu ilegível.
      2. Depois de degradar: fundo real + blur + ruído às vezes se
         combinam de um jeito que apaga ou borra demais o caractere. Mede
         contraste + fill ratio; se ficar fora da faixa, tenta de novo (novo
         sorteio de fundo/blur/ruído, mesma fonte/posição). Se esgotar as
         tentativas, gera uma versão sem blur/ruído -- garante legibilidade
         na grande maioria dos casos (raro caso em que nem isso resolve, sem
         nunca deixar de emitir uma amostra).

    `avisos`: se uma lista for passada, o aviso de legibilidade sub-ótima é
    só ACUMULADO nela (sem print) -- essencial em geração em massa (centenas
    de milhares de amostras, ~10-12% de taxa de aviso), onde imprimir um
    print por ocorrência trava a renderização de output do notebook
    (Jupyter/Kaggle engasga com dezenas de milhares de linhas numa célula só).
    Se `avisos` for None (uso direto/sanity/teste), imprime na hora como antes.
    """
    img_base = _renderizar_base(char, font_path, font_size, canvas_margin, rng, output_size)
    mask_glifo = img_base <= 240

    if not _legivel(img_base, mask_glifo):
        for _ in range(CLF_MAX_TENTATIVAS_MORFO):
            img_try = _renderizar_base(char, font_path, font_size, canvas_margin, rng, output_size)
            mask_try = img_try <= 240
            if _legivel(img_try, mask_try):
                img_base, mask_glifo = img_try, mask_try
                break
        else:
            if fonts_fallback:
                alternativas = [f for f in fonts_fallback if f != font_path]
                rng.shuffle(alternativas)
                for alt in alternativas:
                    img_alt = _renderizar_base(char, alt, font_size, canvas_margin, rng, output_size)
                    mask_alt = img_alt <= 240
                    if _legivel(img_alt, mask_alt):
                        img_base, mask_glifo = img_alt, mask_alt
                        break

    for _ in range(CLF_MAX_TENTATIVAS):
        candidato = _aplicar_fundo_e_degradacoes(img_base, rng)
        if (_contraste_glifo(candidato, mask_glifo) >= CLF_MIN_CONTRASTE and
                CLF_MIN_FILL_RATIO <= _fill_ratio_glifo(candidato, mask_glifo) <= CLF_MAX_FILL_RATIO):
            return candidato

    # Esgotou as tentativas -- abre mão de blur/ruído pra garantir legibilidade
    fallback = _aplicar_fundo_e_degradacoes(img_base, rng, aplicar_blur_ruido=False)
    if (_contraste_glifo(fallback, mask_glifo) < CLF_MIN_CONTRASTE or
            not (CLF_MIN_FILL_RATIO <= _fill_ratio_glifo(fallback, mask_glifo) <= CLF_MAX_FILL_RATIO)):
        msg = f"{char} / {os.path.basename(font_path)} @ {font_size}px"
        if avisos is not None:
            avisos.append(msg)
        else:
            print(f"[WARN] Fora da faixa de legibilidade mesmo apos fallback: {msg}")
    return fallback


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
    avisos = []
    n_falhas = 0

    for char in tqdm(kanjis, desc=f"[{split_name}]"):
        class_dir = os.path.join(output_dir, codepoint_dir(char))
        os.makedirs(class_dir, exist_ok=True)

        for i in range(samples_per_class):
            font_path = rng.choice(fonts)
            font_size = rng.choices(CLASSIFIER_FONT_SIZES, weights=CLASSIFIER_FONT_SIZE_WEIGHTS)[0]
            try:
                img = generate_sample(
                    char=char,
                    font_path=font_path,
                    font_size=font_size,
                    rng=rng,
                    output_size=CLASSIFIER_INPUT_SIZE,
                    canvas_margin=CLASSIFIER_CANVAS_MARGIN,
                    fonts_fallback=fonts,
                    avisos=avisos,
                )
                Image.fromarray(img).save(
                    os.path.join(class_dir, f"{i:03d}.png")
                )
            except Exception as e:
                n_falhas += 1
                if n_falhas <= 20:
                    print(f"[WARN] Falha em {char} com {os.path.basename(font_path)} @ {font_size}px: {e}")

    total = len(kanjis) * samples_per_class
    if avisos:
        print(f"[INFO] [{split_name}] {len(avisos)}/{total} amostras ({100*len(avisos)/max(1,total):.1f}%) "
              f"com legibilidade sub-otima mesmo apos fallback -- primeiras 5: {avisos[:5]}")
    if n_falhas > 20:
        print(f"[AVISO] [{split_name}] {n_falhas} falhas totais na geracao (só as 20 primeiras foram exibidas acima)")


def generate_sanity(kanjis: list, fonts: list, output_dir: str, count: int, seed: int):
    """Gera N amostras aleatórias pra inspeção visual (mix de classes)."""
    rng = random.Random(seed)
    np.random.seed(seed)

    os.makedirs(output_dir, exist_ok=True)

    for i in range(count):
        char = rng.choice(kanjis)
        font_path = rng.choice(fonts)
        font_size = rng.choices(CLASSIFIER_FONT_SIZES, weights=CLASSIFIER_FONT_SIZE_WEIGHTS)[0]
        try:
            img = generate_sample(
                char=char,
                font_path=font_path,
                font_size=font_size,
                rng=rng,
                output_size=CLASSIFIER_INPUT_SIZE,
                canvas_margin=CLASSIFIER_CANVAS_MARGIN,
                fonts_fallback=fonts,
            )
            filename = f"{i:03d}_{codepoint_dir(char)}_{os.path.basename(font_path)[:15]}_{font_size}.png"
            Image.fromarray(img).save(os.path.join(output_dir, filename))
        except Exception as e:
            print(f"[WARN] Falha sanity em {char}: {e}")

    print(f"[INFO] {count} amostras sanity em: {output_dir}")


def build_outros_pools(n1_kanjis: list) -> dict:
    """
    Monta os pools de caracteres não-N1 usados na classe "outros":
      - kanji: qualquer kanji fora da lista N1
      - kana:  hiragana + katakana
      - latim: letras e dígitos ASCII
    "ruido" não tem pool de caracteres — é tratado à parte (fundo vazio/degradado).
    """
    n1_set = set(n1_kanjis)
    todos_kanjis = get_kanjis("")  # sem filtro de nível = todos do kanji.json
    pool_kanji = [k for k in todos_kanjis if k not in n1_set]

    pool_kana = (
        [chr(c) for c in range(0x3041, 0x3097)] +  # hiragana
        [chr(c) for c in range(0x30A1, 0x30F7)]    # katakana
    )

    pool_latim = list(
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!?.,"
    )

    return {"kanji": pool_kanji, "kana": pool_kana, "latim": pool_latim}


def generate_noise_sample(rng: random.Random, output_size: int) -> np.ndarray:
    """
    Amostra "outros" sem nenhum glyph — simula uma bbox do detector que não
    capturou um caractere de verdade (fundo/screentone/arte confundido com texto).
    """
    canvas = np.full((output_size, output_size), 255, dtype=np.uint8)
    img = apply_paper_background(canvas, rng)
    img = apply_blur(img, rng)
    img = apply_noise(img, rng)
    img = apply_brightness_contrast(img, rng)
    img = apply_jpeg(img, rng)
    return img


def generate_outros_split(fonts: list, pools: dict, samples: int,
                          output_dir: str, seed: int, split_name: str):
    """Gera a pasta CLF_OUTROS_LABEL inteira, misturando as subcategorias."""
    rng = random.Random(seed)
    np.random.seed(seed)

    class_dir = os.path.join(output_dir, CLF_OUTROS_LABEL)
    os.makedirs(class_dir, exist_ok=True)

    categorias = (
        ["kanji"] * round(samples * CLF_OUTROS_PROP_KANJI) +
        ["kana"]  * round(samples * CLF_OUTROS_PROP_KANA) +
        ["latim"] * round(samples * CLF_OUTROS_PROP_LATIM) +
        ["ruido"] * round(samples * CLF_OUTROS_PROP_RUIDO)
    )
    rng.shuffle(categorias)

    avisos = []
    n_falhas = 0
    for i, cat in enumerate(tqdm(categorias, desc=f"[{split_name}-outros]")):
        font_path = rng.choice(fonts)
        font_size = rng.choices(CLASSIFIER_FONT_SIZES, weights=CLASSIFIER_FONT_SIZE_WEIGHTS)[0]
        try:
            if cat == "ruido":
                img = generate_noise_sample(rng, CLASSIFIER_INPUT_SIZE)
            else:
                char = rng.choice(pools[cat])
                img = generate_sample(
                    char=char, font_path=font_path, font_size=font_size,
                    rng=rng, output_size=CLASSIFIER_INPUT_SIZE,
                    canvas_margin=CLASSIFIER_CANVAS_MARGIN,
                    fonts_fallback=fonts,
                    avisos=avisos,
                )
            Image.fromarray(img).save(os.path.join(class_dir, f"{i:04d}.png"))
        except Exception as e:
            n_falhas += 1
            if n_falhas <= 20:
                print(f"[WARN] Falha outros ({cat}): {e}")

    if avisos:
        print(f"[INFO] [{split_name}-outros] {len(avisos)}/{len(categorias)} amostras "
              f"({100*len(avisos)/max(1,len(categorias)):.1f}%) com legibilidade sub-otima -- "
              f"primeiras 5: {avisos[:5]}")
    if n_falhas > 20:
        print(f"[AVISO] [{split_name}-outros] {n_falhas} falhas totais (só as 20 primeiras foram exibidas acima)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="Gera só as primeiras N classes (para testar rápido)")
    parser.add_argument("--sanity-only", action="store_true",
                        help="Só regenera o modo sanity, sem train/val")
    parser.add_argument("--skip-sanity", action="store_true",
                        help="Pula geração do sanity")
    parser.add_argument("--skip-outros", action="store_true",
                        help="Pula geração da classe 'outros' (rejeição de não-N1)")
    args = parser.parse_args()

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

    if not args.skip_sanity:
        print(f"\n[INFO] Gerando {CLASSIFIER_SANITY_COUNT} amostras sanity...")
        generate_sanity(kanjis, fonts, CLASSIFIER_SANITY_DIR,
                        CLASSIFIER_SANITY_COUNT, CLASSIFIER_SEED_TRAIN)

    if args.sanity_only:
        print("[INFO] Modo sanity-only. Encerrando.")
        return

    print(f"\n[INFO] Gerando treino: {CLASSIFIER_SAMPLES_TRAIN} amostras × {len(kanjis)} classes = "
          f"{CLASSIFIER_SAMPLES_TRAIN * len(kanjis)} crops")
    generate_split(kanjis, fonts, CLASSIFIER_SAMPLES_TRAIN,
                   CLASSIFIER_TRAIN_DIR, CLASSIFIER_SEED_TRAIN, "train")

    print(f"\n[INFO] Gerando validação: {CLASSIFIER_SAMPLES_VAL} amostras × {len(kanjis)} classes = "
          f"{CLASSIFIER_SAMPLES_VAL * len(kanjis)} crops")
    generate_split(kanjis, fonts, CLASSIFIER_SAMPLES_VAL,
                   CLASSIFIER_VAL_DIR, CLASSIFIER_SEED_VAL, "val")

    if not args.skip_outros:
        print(f"\n[INFO] Montando pools da classe '{CLF_OUTROS_LABEL}'...")
        pools = build_outros_pools(kanjis)
        print(f"[INFO] Pool kanji não-N1: {len(pools['kanji'])} | kana: {len(pools['kana'])} | latim: {len(pools['latim'])}")

        print(f"\n[INFO] Gerando treino '{CLF_OUTROS_LABEL}': {CLF_OUTROS_SAMPLES_TRAIN} crops")
        generate_outros_split(fonts, pools, CLF_OUTROS_SAMPLES_TRAIN,
                             CLASSIFIER_TRAIN_DIR, CLASSIFIER_SEED_TRAIN, "train")

        print(f"\n[INFO] Gerando validação '{CLF_OUTROS_LABEL}': {CLF_OUTROS_SAMPLES_VAL} crops")
        generate_outros_split(fonts, pools, CLF_OUTROS_SAMPLES_VAL,
                             CLASSIFIER_VAL_DIR, CLASSIFIER_SEED_VAL, "val")

    print("\n[INFO] Geração completa!")


if __name__ == "__main__":
    main()
