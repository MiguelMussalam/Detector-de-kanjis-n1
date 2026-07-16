"""
corpus_validate.py
===================
Roda o pipeline completo (detector + classificador) sobre o ground truth
full-corpus (data/corpus_validation/ground_truth_full.json, construido por
`src/helper/manga109_corpus.py`) e mede recall agregado + por volume.

Complementa `src/helper/benchmark.py` (4 paginas curadas manualmente, baixo
ruido) com uma medida em escala real (ate ~10.600 paginas) -- os dois
convivem, nenhum substitui o outro.

Antes de rodar o corpus inteiro, medir tempo num subconjunto pequeno
(`--limit-volumes 3`) e extrapolar -- `Pipeline.predict()` faz um forward do
classificador por deteccao, sem batching, e uma pagina de manga pode ter
100-300+ caracteres.

Uso:
    python -m src.helper.corpus_validate --limit-volumes 3
    python -m src.helper.corpus_validate
"""

import argparse
import json
import os
import random
import time

import cv2
from tqdm import tqdm

from config import ROOT_DIR, MANGA109_IMAGES
from src.pipeline.inference import Pipeline
from src.helper.benchmark import avaliar_pagina
from src.helper.manga109_corpus import GT_PATH as GT_PATH_DEFAULT

CORPUS_DIR = os.path.join(ROOT_DIR, "data", "corpus_validation")
RESULT_PATH = os.path.join(CORPUS_DIR, "resultado_full.json")
TOTAL_PAGINAS_MANGA109 = 10602  # referencia pra extrapolar tempo (ver plano)


def _filtrar_paginas(gt: dict, volumes=None, sample_por_volume=None, seed=42) -> list:
    rng = random.Random(seed)
    paginas = gt["paginas"]

    if volumes:
        volumes_set = set(volumes)
        paginas = [p for p in paginas if p["origem"]["volume"] in volumes_set]

    if sample_por_volume:
        por_volume = {}
        for p in paginas:
            por_volume.setdefault(p["origem"]["volume"], []).append(p)
        amostradas = []
        for ps in por_volume.values():
            rng.shuffle(ps)
            amostradas.extend(ps[:sample_por_volume])
        paginas = amostradas

    return paginas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground-truth", default=GT_PATH_DEFAULT)
    parser.add_argument("--volumes", type=str, default=None, help="lista separada por virgula")
    parser.add_argument("--limit-volumes", type=int, default=None)
    parser.add_argument("--sample-paginas-por-volume", type=int, default=None)
    parser.add_argument("--out", default=RESULT_PATH)
    args = parser.parse_args()

    with open(args.ground_truth, encoding="utf-8") as f:
        gt = json.load(f)

    volumes = args.volumes.split(",") if args.volumes else None
    if args.limit_volumes and not volumes:
        todos = sorted(set(p["origem"]["volume"] for p in gt["paginas"]))
        volumes = todos[:args.limit_volumes]

    paginas = _filtrar_paginas(
        gt, volumes=volumes, sample_por_volume=args.sample_paginas_por_volume
    )
    print(f"[INFO] Avaliando {len(paginas)} paginas...")

    print("[INFO] Carregando pipeline...")
    pipeline = Pipeline()

    por_volume = {}
    hits_total, esperado_total = 0, 0
    deteccoes_total, outros_total = 0, 0
    inicio = time.time()

    for pagina in tqdm(paginas, desc="paginas"):
        origem = pagina["origem"]
        img_path = os.path.join(MANGA109_IMAGES, origem["volume"], f"{origem['pagina']:03d}.jpg")
        frame = cv2.imread(img_path)
        if frame is None:
            continue

        deteccoes = pipeline.predict(frame)
        n_outros = sum(1 for d in deteccoes if d.codepoint == "OUTROS")
        deteccoes_total += len(deteccoes)
        outros_total += n_outros

        resultado = avaliar_pagina(deteccoes, pagina["linhas"])
        hits_total += resultado["hits"]
        esperado_total += resultado["esperado"]

        vol_stats = por_volume.setdefault(origem["volume"], {"hits": 0, "esperado": 0})
        vol_stats["hits"] += resultado["hits"]
        vol_stats["esperado"] += resultado["esperado"]

    duracao = time.time() - inicio
    seg_por_pagina = duracao / max(1, len(paginas))

    resumo = {
        "n_paginas": len(paginas),
        "duracao_segundos": duracao,
        "segundos_por_pagina": seg_por_pagina,
        "extrapolado_corpus_completo_horas": seg_por_pagina * TOTAL_PAGINAS_MANGA109 / 3600,
        "deteccoes_total": deteccoes_total,
        "outros_total": outros_total,
        "outros_pct": 100 * outros_total / max(1, deteccoes_total),
        "hits_total": hits_total,
        "esperado_total": esperado_total,
        "recall_pct": 100 * hits_total / max(1, esperado_total),
        "por_volume": {
            vol: {**s, "recall_pct": 100 * s["hits"] / max(1, s["esperado"])}
            for vol, s in por_volume.items()
        },
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(resumo, f, ensure_ascii=False, indent=2)

    print(f"\n[RESUMO] {len(paginas)} paginas, {seg_por_pagina:.2f}s/pagina "
          f"(extrapolado p/ {TOTAL_PAGINAS_MANGA109} paginas: "
          f"{resumo['extrapolado_corpus_completo_horas']:.1f}h)")
    print(f"[RESUMO] {deteccoes_total} deteccoes, {outros_total} "
          f"({resumo['outros_pct']:.1f}%) classificadas como OUTROS")
    print(f"[RESUMO] recall = {hits_total}/{esperado_total} ({resumo['recall_pct']:.1f}%)")
    print(f"[INFO] Detalhado salvo em {args.out}")


if __name__ == "__main__":
    main()
