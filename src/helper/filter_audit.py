"""
filter_audit.py
================
Auditoria visual + quantitativa de cada filtro de degradação usado em
generate_crops.py, isolado dos demais. Pra cada filtro, varre um range de
severidade (do zero até visivelmente extremo, além do range de produção
atual) e gera:

  - data/benchmark/filters/<filtro>/grid.png -- grade visual (linhas =
    severidade, colunas = amostras aleatórias), pra inspeção visual de onde
    cada filtro fica fraco demais (não muda nada) ou extremo demais (destrói
    legibilidade).
  - data/benchmark/filters/<filtro>/metricas.csv -- acurácia/confiança/%OUTROS
    do classificador atual em cada nível de severidade (mesmo método usado
    pra medir o impacto da corrosão nesta sessão, generalizado).

Cada filtro é isolado dos demais (todos os outros desligados) -- só assim dá
pra atribuir o efeito visto a UM filtro específico, não a uma combinação.

Uso:
    python -m src.helper.filter_audit                  # todos os filtros
    python -m src.helper.filter_audit --filtro erosao   # so um
"""

import argparse
import csv
import io
import os
import random

import cv2
import numpy as np
import torch
from PIL import Image, ImageFilter
from scipy.ndimage import grey_dilation, grey_erosion

from config import (
    ROOT_DIR, CLASSIFIER_INPUT_SIZE, CLASSIFIER_CANVAS_MARGIN,
    CLASSIFIER_FONT_SIZES, CLASSIFIER_FONT_SIZE_WEIGHTS,
    CLF_WEIGHTS_PATH, CLF_OUTROS_LABEL,
    CLF_ROTATION_MAX, CLF_TRANSLATE_MAX, CLF_BLUR_SIGMA_MAX, CLF_NOISE_STD_MAX,
    CLF_BRIGHTNESS_RANGE, CLF_CONTRAST_RANGE, CLF_JPEG_QUALITY_MIN, CLF_MORFO_K_MAX,
)
from src.helper.kanjis import get_kanjis
from src.helper.fonts import get_fonts_list
from src.classifier.generate_crops import render_glyph, apply_paper_background, _fill_ratio_glifo
from src.classifier.dataset import build_transform
from src.pipeline.inference import load_classifier

FILTERS_DIR = os.path.join(ROOT_DIR, "data", "benchmark", "filters")

# Valor de producao atual (config.py) pra cada filtro -- referencia impressa
# no resumo, pra ver se o range de producao fica numa zona segura ou ja
# proximo do "penhasco" de acuracia.
CONFIG_ATUAL = {
    "erosao": CLF_MORFO_K_MAX, "dilatacao": CLF_MORFO_K_MAX,
    "rotacao": CLF_ROTATION_MAX, "translacao": CLF_TRANSLATE_MAX,
    "blur": CLF_BLUR_SIGMA_MAX, "ruido": CLF_NOISE_STD_MAX,
    "brilho": CLF_BRIGHTNESS_RANGE, "contraste": CLF_CONTRAST_RANGE,
    "jpeg": CLF_JPEG_QUALITY_MIN,
}


def _base_limpa(char, font_path, font_size):
    """Render puro, sem nenhuma degradação -- ponto de partida neutro pra isolar um filtro por vez."""
    canvas_size = int(font_size * (1 + 2 * CLASSIFIER_CANVAS_MARGIN))
    return render_glyph(char, font_path, font_size, canvas_size)


def _ajustar_tamanho(img, output_size):
    h, w = img.shape
    y0 = max(0, (h - output_size) // 2)
    x0 = max(0, (w - output_size) // 2)
    crop = img[y0:y0 + output_size, x0:x0 + output_size]
    if crop.shape != (output_size, output_size):
        crop = np.array(Image.fromarray(crop).resize((output_size, output_size), Image.BILINEAR))
    return crop


# --- filtros isolados, severidade explicita (nao amostrada de um range) ---

def _erosao(img, k, rng):
    return grey_dilation(img, size=(k, k)) if k > 0 else img  # "erode" nesta convencao (preto/0 em branco/255)


def _dilatacao(img, k, rng):
    return grey_erosion(img, size=(k, k)) if k > 0 else img  # "dilate" nesta convencao


def _rotacao(img, graus, rng):
    if graus == 0:
        return img
    return np.array(Image.fromarray(img).rotate(graus, resample=Image.BILINEAR, fillcolor=255))


def _translacao(img, frac, rng):
    if frac == 0:
        return img
    h, w = img.shape
    max_dx = int(w * frac)
    max_dy = int(h * frac)
    dx = max_dx * rng.choice([-1, 1])
    dy = max_dy * rng.choice([-1, 1])
    # recorte precisa ser MENOR que o canvas, reservando a folga do
    # deslocamento -- mesmo fix aplicado em generate_crops.py (ver
    # apply_translate_and_crop): sem isso a janela nao tem espaco pra deslizar.
    crop_size = min(h - 2 * max_dy, w - 2 * max_dx)
    cx, cy = w // 2 + dx, h // 2 + dy
    x0 = max(0, min(w - crop_size, cx - crop_size // 2))
    y0 = max(0, min(h - crop_size, cy - crop_size // 2))
    return img[y0:y0 + crop_size, x0:x0 + crop_size]


def _blur(img, sigma, rng):
    if sigma == 0:
        return img
    return np.array(Image.fromarray(img).filter(ImageFilter.GaussianBlur(radius=sigma)))


def _ruido(img, std_frac, rng):
    if std_frac == 0:
        return img
    noise = np.random.normal(0, std_frac * 255, img.shape)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def _brilho(img, delta_frac, rng):
    if delta_frac == 0:
        return img
    delta = delta_frac * 255 * rng.choice([-1, 1])
    return np.clip(img.astype(np.float32) + delta, 0, 255).astype(np.uint8)


def _contraste(img, fator_delta, rng):
    if fator_delta == 0:
        return img
    factor = 1.0 + fator_delta * rng.choice([-1, 1])
    mean = img.astype(np.float32).mean()
    return np.clip((img.astype(np.float32) - mean) * factor + mean, 0, 255).astype(np.uint8)


def _jpeg(img, qualidade, rng):
    if qualidade >= 100:
        return img
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=qualidade)
    buf.seek(0)
    return np.array(Image.open(buf))


# (nome, severidades, aplica_antes_do_fundo, funcao)
FILTROS = [
    ("erosao",     [0, 1, 2, 3, 4, 5, 6, 8],              True,  _erosao),
    ("dilatacao",  [0, 1, 2, 3, 4, 5, 6, 8],              True,  _dilatacao),
    ("rotacao",    [0, 3, 6, 10, 15, 20, 30],             True,  _rotacao),
    ("translacao", [0.0, 0.05, 0.10, 0.15, 0.20, 0.30],   True,  _translacao),
    ("blur",       [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0],   False, _blur),
    ("ruido",      [0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12], False, _ruido),
    ("brilho",     [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],        False, _brilho),
    ("contraste",  [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],        False, _contraste),
    ("jpeg",       [100, 90, 70, 50, 30, 15, 5],          False, _jpeg),
]


def gerar_amostra(severidade, aplica_antes_do_fundo, func_filtro, char, font_path, font_size, rng):
    img = _base_limpa(char, font_path, font_size)

    if aplica_antes_do_fundo:
        img = func_filtro(img, severidade, rng)
        img = _ajustar_tamanho(img, CLASSIFIER_INPUT_SIZE)
        mask = img <= 240
        img_final = apply_paper_background(img, rng)
    else:
        img = _ajustar_tamanho(img, CLASSIFIER_INPUT_SIZE)
        mask = img <= 240
        img_final = apply_paper_background(img, rng)
        img_final = func_filtro(img_final, severidade, rng)

    return img_final, mask


def montar_grid(imagens_por_severidade, severidades, out_path, cell=90, label_w=70):
    n_col = max(len(imgs) for imgs in imagens_por_severidade)
    sheet = np.full((len(severidades) * cell, label_w + n_col * cell, 3), 255, dtype=np.uint8)
    for i, sev in enumerate(severidades):
        for j, img in enumerate(imagens_por_severidade[i]):
            thumb = cv2.resize(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), (cell, cell))
            y0 = i * cell
            x0 = label_w + j * cell
            sheet[y0:y0 + cell, x0:x0 + cell] = thumb
        cv2.putText(sheet, str(sev), (5, i * cell + cell // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(out_path, sheet)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filtro", type=str, default=None, help="roda so um filtro (ver FILTROS)")
    parser.add_argument("--n-grid", type=int, default=10, help="amostras mostradas na grade visual")
    parser.add_argument("--n-metricas", type=int, default=40, help="amostras usadas nas metricas do classificador")
    args = parser.parse_args()

    n1 = get_kanjis("n1")
    fonts = get_fonts_list()
    rng = random.Random(42)

    print("[INFO] Carregando classificador...")
    model, classes = load_classifier(CLF_WEIGHTS_PATH, device="cpu")
    model.eval()
    transform = build_transform()
    idx_outros = classes.index(CLF_OUTROS_LABEL)

    filtros = [f for f in FILTROS if args.filtro is None or f[0] == args.filtro]
    if not filtros:
        raise ValueError(f"Filtro '{args.filtro}' desconhecido. Opcoes: {[f[0] for f in FILTROS]}")

    resumo_linhas = []

    for nome, severidades, antes_fundo, func in filtros:
        print(f"\n[INFO] Filtro: {nome}")
        out_dir = os.path.join(FILTERS_DIR, nome)
        os.makedirs(out_dir, exist_ok=True)

        # Amostras fixas (mesmo char/fonte/tamanho) por coluna -- sorteadas
        # uma unica vez, reusadas em TODAS as severidades. Sem isso, cada
        # severidade comparava kanji/fonte diferentes entre si, tornando
        # efeitos sutis (ex: translacao) impossiveis de perceber visualmente
        # -- a linha de uma severidade nunca era o "mesmo kanji" da anterior.
        amostras_fixas = [(
            rng.choice(n1),
            rng.choice(fonts),
            rng.choices(CLASSIFIER_FONT_SIZES, weights=CLASSIFIER_FONT_SIZE_WEIGHTS)[0],
        ) for _ in range(args.n_metricas)]

        imagens_para_grid = []
        linhas_csv = []

        for sev in severidades:
            imagens_sev = []
            for i, (char, font_path, font_size) in enumerate(amostras_fixas):
                # RNG proprio por coluna, re-semeado igual a cada severidade --
                # garante que a UNICA coisa que muda de uma severidade pra
                # outra, pra essa coluna, seja a severidade em si (direcao/sinal
                # aleatorios do filtro e escolha de fundo ficam identicos entre
                # severidades, isolando so o efeito da severidade).
                rng_amostra = random.Random(9000 + i)
                np.random.seed(9000 + i)

                img_final, mask = gerar_amostra(sev, antes_fundo, func, char, font_path, font_size, rng_amostra)
                if i < args.n_grid:
                    imagens_sev.append(img_final)

                fill_ratio = _fill_ratio_glifo(img_final, mask)
                pil = Image.fromarray(img_final)
                x = transform(pil).unsqueeze(0)
                with torch.no_grad():
                    probs = torch.softmax(model(x), dim=1)[0]
                    pred_idx = int(probs.argmax())
                    conf = float(probs[pred_idx])
                pred_classe = classes[pred_idx]
                correto = (pred_classe != CLF_OUTROS_LABEL and chr(int(pred_classe.replace("U+", ""), 16)) == char)

                linhas_csv.append({
                    "severidade": sev, "char": char, "fill_ratio": fill_ratio,
                    "correto": correto, "outros": pred_idx == idx_outros, "conf": conf,
                })
            imagens_para_grid.append(imagens_sev)

        montar_grid(imagens_para_grid, severidades, os.path.join(out_dir, "grid.png"))

        with open(os.path.join(out_dir, "metricas.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(linhas_csv[0].keys()))
            w.writeheader()
            w.writerows(linhas_csv)

        print(f"{'severidade':>12} {'acuracia':>10} {'% OUTROS':>10} {'fill_ratio':>12} {'conf':>8}")
        for sev in severidades:
            subset = [r for r in linhas_csv if r["severidade"] == sev]
            acc = sum(r["correto"] for r in subset) / len(subset)
            pct_outros = sum(r["outros"] for r in subset) / len(subset)
            fr = sum(r["fill_ratio"] for r in subset) / len(subset)
            conf = sum(r["conf"] for r in subset) / len(subset)
            resumo_linhas.append((nome, sev, acc, pct_outros, fr, conf))
            print(f"{str(sev):>12} {100*acc:>9.1f}% {100*pct_outros:>9.1f}% {fr:>12.3f} {conf:>8.3f}")

        print(f"[INFO] grid.png + metricas.csv em {out_dir}")

    resumo_path = os.path.join(FILTERS_DIR, "resumo.md")
    with open(resumo_path, "w", encoding="utf-8") as f:
        f.write("# Auditoria de filtros -- resumo\n\n")
        for nome, severidades, _, _ in filtros:
            valor_atual = CONFIG_ATUAL.get(nome)
            f.write(f"## {nome} (config.py atual: {valor_atual})\n\n")
            f.write("| severidade | acuracia | % OUTROS | fill_ratio | confianca |\n")
            f.write("|---|---|---|---|---|\n")
            for n, sev, acc, pct_outros, fr, conf in resumo_linhas:
                if n != nome:
                    continue
                f.write(f"| {sev} | {100*acc:.1f}% | {100*pct_outros:.1f}% | {fr:.3f} | {conf:.3f} |\n")
            f.write("\n")
    print(f"\n[INFO] Resumo consolidado em {resumo_path}")


if __name__ == "__main__":
    main()
