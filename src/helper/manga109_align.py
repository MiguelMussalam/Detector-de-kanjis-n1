"""
manga109_align.py
==================
Gera dado real (crop + rótulo) para fine-tuning do classificador, sem
anotação manual — deduzindo o rótulo de cada caractere a partir da
transcrição das falas do Manga109 (<text>) cruzada com as bboxes que o
NOSSO detector (já treinado) encontra na página inteira.

Método (verificado visualmente antes de rodar em escala — ver conversa):
    1. Roda o detector na página inteira (não por linha — recorte isolado
       de uma bbox de fala fica pequeno demais e o detector não acha nada).
    2. Para cada <text> (bbox da fala + string transcrita):
       - Limpa a string (remove pontuação/símbolos que o detector não caixa).
       - Filtra as bboxes do detector cujo centro cai dentro da bbox da fala.
       - Só aceita se a contagem bater (nº de bboxes == nº de caracteres).
       - Agrupa as bboxes aceitas por coluna (x) e ordena topo->baixo dentro
         da coluna, colunas da direita pra esquerda (leitura vertical
         japonesa). Bboxes de fala mais largas que altas (texto horizontal)
         são ignoradas — caso raro nesse corpus, não vale a complexidade agora.
       - Zipa bbox_ordenada[i] <-> string_limpa[i].
    3. Salva o crop (com o mesmo padding usado na inferência real, para bater
       com o que o classificador vê em produção) em:
         - data/classifier_real/{train,val}/U+XXXX/  se o caractere é kanji N1
         - data/classifier_real/{train,val}/OUTROS/  caso contrário (kana,
           kanji fora do N1, etc — negativos REAIS para a classe de rejeição)
       Split train/val por VOLUME (ver MANGA109_ALIGN_VAL_VOLUMES), não por
       crop, para não vazar página de val no treino.

Uso:
    python -m src.helper.manga109_align                  # todos os volumes
    python -m src.helper.manga109_align --limit-volumes 5 # so os N primeiros (teste)
"""

import argparse
import os
import xml.etree.ElementTree as ET
from glob import glob

import cv2
from PIL import Image
from tqdm import tqdm

from config import (
    MANGA109_ANNOTATIONS, MANGA109_IMAGES,
    MANGA109_ALIGN_TRAIN_DIR, MANGA109_ALIGN_VAL_DIR, MANGA109_ALIGN_VAL_VOLUMES,
    MANGA109_ALIGN_COLUNA_TOL_FRAC, MANGA109_ALIGN_MAX_COLUNAS, MANGA109_ALIGN_PONTUACAO,
    MANGA109_ALIGN_COLUNA_ISOLADA_MIN_VIZINHA, MANGA109_ALIGN_GAP_Y_FATOR,
    DETECTOR_WEIGHTS_PATH, PIPELINE_DET_CONF, PIPELINE_DET_IOU, PIPELINE_DET_MAX_DET,
    PIPELINE_BBOX_PADDING,
    CLF_OUTROS_LABEL,
)
from src.helper.kanjis import get_kanjis
from src.pipeline.inference import load_detector, _expandir_bbox


# ---------------------------------------------------------------------------
# Limpeza e alinhamento de uma linha
# ---------------------------------------------------------------------------

def limpar_string(s: str) -> str:
    return "".join(c for c in (s or "") if c not in MANGA109_ALIGN_PONTUACAO)


def _dentro(box, x1, y1, x2, y2) -> bool:
    bx1, by1, bx2, by2 = box
    cx, cy = (bx1 + bx2) / 2, (by1 + by2) / 2
    return x1 <= cx <= x2 and y1 <= cy <= y2


def agrupar_colunas(boxes: list) -> list:
    """
    Agrupa bboxes em colunas de leitura vertical (direita -> esquerda).

    Tolerância adaptativa: fração da largura MEDIANA das boxes dessa linha
    específica, em vez de um valor fixo em pixels. Um valor fixo (15px) se
    mostrou frágil em painéis com colunas compactas — já vimos casos reais
    com colunas a ~13-16px de distância uma da outra, quase do tamanho de
    um caractere, onde a tolerância fixa junta ou separa colunas errado.

    Cada coluna retornada já vem ordenada topo->baixo.
    """
    if not boxes:
        return []

    larguras = sorted(b[2] - b[0] for b in boxes)
    largura_mediana = larguras[len(larguras) // 2]
    tolerancia = max(6.0, largura_mediana * MANGA109_ALIGN_COLUNA_TOL_FRAC)

    boxes_por_x = sorted(boxes, key=lambda b: -(b[0] + b[2]) / 2)
    colunas = []
    for b in boxes_por_x:
        cx = (b[0] + b[2]) / 2
        for col in colunas:
            if abs(col["cx"] - cx) < tolerancia:
                col["boxes"].append(b)
                # recalcula o centro da coluna como media corrente, para nao
                # deixar o critério de comparação "derivar" com boxes novas
                col["cx"] = sum((bb[0] + bb[2]) / 2 for bb in col["boxes"]) / len(col["boxes"])
                break
        else:
            colunas.append({"cx": cx, "boxes": [b]})

    for col in colunas:
        col["boxes"].sort(key=lambda b: (b[1] + b[3]) / 2)

    return colunas


def ordenar_leitura(boxes: list) -> list:
    """Agrupa em colunas (ver agrupar_colunas) e concatena na ordem de leitura."""
    ordenado = []
    for col in agrupar_colunas(boxes):
        ordenado.extend(col["boxes"])
    return ordenado


def colunas_sao_confiaveis(colunas: list) -> bool:
    """
    Detecta duas falhas de alinhamento que passam pela checagem de contagem
    (nº de boxes == nº de caracteres) porque um caractere perdido pelo
    detector é "compensado" por uma caixa espúria em outro lugar da mesma
    linha -- o total bate, mas o zip bbox<->char sai deslocado a partir do
    ponto do erro. Achados em revisão manual (ver conversa):

      1. Coluna isolada de 1 caixa "encaixada" entre colunas bem maiores --
         normalmente é um artefato (pontuação/ruído que o detector caixou
         por engano), não uma coluna de texto real.
      2. Salto de Y anormal dentro de uma coluna -- caracteres vizinhos numa
         coluna vertical ficam bem colados; um vão muito maior que a altura
         típica das boxes ali sinaliza um caractere que o detector não achou
         naquele ponto.
    """
    if len(colunas) <= 1:
        return True  # nada para comparar

    tamanhos = [len(col["boxes"]) for col in colunas]
    maior = max(tamanhos)
    for tamanho in tamanhos:
        if tamanho == 1 and maior >= MANGA109_ALIGN_COLUNA_ISOLADA_MIN_VIZINHA:
            return False

    for col in colunas:
        boxes = col["boxes"]
        if len(boxes) < 2:
            continue
        alturas = sorted(b[3] - b[1] for b in boxes)
        altura_mediana = alturas[len(alturas) // 2]
        limite_gap = altura_mediana * MANGA109_ALIGN_GAP_Y_FATOR
        for i in range(1, len(boxes)):
            gap = boxes[i][1] - boxes[i - 1][3]
            if gap > limite_gap:
                return False

    return True


def alinhar_pagina(frame_bgr, text_elements: list, todos_boxes: list) -> list:
    """
    Recebe os elementos <text> de uma pagina e os boxes ja detectados
    (rodados uma vez na pagina inteira). Retorna lista de (bbox, char).
    """
    pares = []
    for text_el in text_elements:
        s_limpo = limpar_string(text_el.text)
        if not s_limpo:
            continue

        x1 = int(text_el.get("xmin")); y1 = int(text_el.get("ymin"))
        x2 = int(text_el.get("xmax")); y2 = int(text_el.get("ymax"))

        # Texto horizontal (bbox mais largo que alto) — pula, foco em vertical
        if (x2 - x1) > (y2 - y1):
            continue

        boxes_dentro = [b for b in todos_boxes if _dentro(b, x1, y1, x2, y2)]
        if len(boxes_dentro) != len(s_limpo):
            continue

        colunas = agrupar_colunas(boxes_dentro)

        # Linhas com muitas colunas sao mais propensas a erro de agrupamento
        # (colunas proximas demais, tolerancia ambigua) -- prefere perder
        # rendimento a aceitar dado errado. Verificado empiricamente: uma
        # linha de 31 caracteres/4+ colunas produziu rotulo errado mesmo com
        # a contagem batendo; uma de 12 caracteres/3 colunas funcionou.
        if len(colunas) > MANGA109_ALIGN_MAX_COLUNAS:
            continue

        if not colunas_sao_confiaveis(colunas):
            continue

        boxes_ordenados = []
        for col in colunas:
            boxes_ordenados.extend(col["boxes"])
        pares.extend(zip(boxes_ordenados, s_limpo))

    return pares


# ---------------------------------------------------------------------------
# Processamento de um volume inteiro
# ---------------------------------------------------------------------------

def processar_volume(nome_volume: str, detector, n1_set: set) -> dict:
    """Processa todas as paginas de um volume. Retorna contagem por categoria."""
    xml_path = os.path.join(MANGA109_ANNOTATIONS, f"{nome_volume}.xml")
    tree = ET.parse(xml_path)
    root = tree.getroot()

    eh_val = nome_volume in MANGA109_ALIGN_VAL_VOLUMES
    out_dir = MANGA109_ALIGN_VAL_DIR if eh_val else MANGA109_ALIGN_TRAIN_DIR

    stats = {"n1": 0, "outros": 0, "paginas": 0}

    for page in root.iter("page"):
        idx = int(page.get("index"))
        img_path = os.path.join(MANGA109_IMAGES, nome_volume, f"{idx:03d}.jpg")
        if not os.path.exists(img_path):
            continue

        frame = cv2.imread(img_path)
        if frame is None:
            continue
        h_frame, w_frame = frame.shape[:2]

        results = detector(
            frame, conf=PIPELINE_DET_CONF, iou=PIPELINE_DET_IOU,
            agnostic_nms=True, max_det=PIPELINE_DET_MAX_DET, verbose=False,
        )[0]
        todos_boxes = [tuple(b.xyxy[0].tolist()) for b in results.boxes]

        text_elements = list(page.iter("text"))
        pares = alinhar_pagina(frame, text_elements, todos_boxes)

        frame_rgb = Image.fromarray(frame[:, :, ::-1])
        for i, (box, char) in enumerate(pares):
            x1, y1, x2, y2 = box
            xe1, ye1, xe2, ye2 = _expandir_bbox(
                x1, y1, x2, y2, PIPELINE_BBOX_PADDING, w_frame, h_frame
            )
            crop = frame_rgb.crop((xe1, ye1, xe2, ye2)).convert("L")

            if char in n1_set:
                classe_dir = f"U+{ord(char):04X}"
                stats["n1"] += 1
            else:
                classe_dir = CLF_OUTROS_LABEL
                stats["outros"] += 1

            dest = os.path.join(out_dir, classe_dir)
            os.makedirs(dest, exist_ok=True)
            crop.save(os.path.join(dest, f"{nome_volume}_{idx:03d}_{i:03d}.png"))

        stats["paginas"] += 1

    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit-volumes", type=int, default=None,
                        help="Processa so os N primeiros volumes (para teste rapido)")
    args = parser.parse_args()

    print("[INFO] Carregando detector...")
    detector = load_detector(DETECTOR_WEIGHTS_PATH)
    n1_set = set(get_kanjis("n1"))

    volumes = sorted(
        os.path.splitext(os.path.basename(f))[0]
        for f in glob(os.path.join(MANGA109_ANNOTATIONS, "*.xml"))
    )
    if args.limit_volumes:
        volumes = volumes[:args.limit_volumes]

    print(f"[INFO] Processando {len(volumes)} volumes...")
    total = {"n1": 0, "outros": 0, "paginas": 0}
    for nome in tqdm(volumes, desc="volumes"):
        stats = processar_volume(nome, detector, n1_set)
        for k in total:
            total[k] += stats[k]

    print(f"\n[INFO] Paginas processadas: {total['paginas']}")
    print(f"[INFO] Crops N1 gerados:    {total['n1']}")
    print(f"[INFO] Crops OUTROS gerados: {total['outros']}")
    print(f"[INFO] Saida em: {MANGA109_ALIGN_TRAIN_DIR} / {MANGA109_ALIGN_VAL_DIR}")


if __name__ == "__main__":
    main()
