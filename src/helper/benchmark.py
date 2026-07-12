"""
benchmark.py
============
Roda o pipeline completo (detector + classificador) na página de referência
fixa (data/benchmark/pagina_teste.jpg + ground_truth.json) e compara contra
uma verdade fundamental que vem DIRETO da anotação oficial do Manga109 (bbox
de linha + transcrição), sem depender da nossa própria heurística de
alinhamento por caractere (`manga109_align.py`) -- essa heurística tem erro
residual conhecido (já achamos casos errados nela), então não serve como
verdade fundamental, só para gerar negativos do OUTROS.

Manga109 não anota por caractere, só por linha de diálogo inteira. Por isso a
checagem aqui é por REGIÃO: um kanji N1 esperado numa linha conta como
"encontrado" se alguma detecção do pipeline cair dentro da bbox da linha com
o mesmo kanji previsto -- é recall por presença na região, não confirmação de
posição exata nem de multiplicidade. Ver "metodologia"/"exclusoes" no próprio
ground_truth.json para o que foi conferido manualmente e o que foi descartado.

Uso:
    python -m src.helper.benchmark
"""

import json
import os

import cv2

from config import ROOT_DIR
from src.pipeline.inference import Pipeline
from src.helper.manga109_align import _dentro

BENCHMARK_DIR = os.path.join(ROOT_DIR, "data", "benchmark")
IMG_PATH = os.path.join(BENCHMARK_DIR, "pagina_teste.jpg")
GT_PATH = os.path.join(BENCHMARK_DIR, "ground_truth.json")


def main():
    with open(GT_PATH, encoding="utf-8") as f:
        gt = json.load(f)

    frame = cv2.imread(IMG_PATH)
    if frame is None:
        raise FileNotFoundError(f"Nao foi possivel ler {IMG_PATH}")

    print("[INFO] Carregando pipeline...")
    pipeline = Pipeline()

    print(f"[INFO] Rodando em {IMG_PATH} ({gt['origem']['volume']} pag {gt['origem']['pagina']})...")
    deteccoes = pipeline.predict(frame)

    total_outros = sum(1 for d in deteccoes if d.codepoint == "OUTROS")
    print(f"\n[CONTEXTO] {len(deteccoes)} deteccoes totais na pagina, "
          f"{total_outros} ({100*total_outros/max(1,len(deteccoes)):.1f}%) classificadas como OUTROS")

    hits, total = 0, 0
    print(f"\n[VERDADE FUNDAMENTAL] {len(gt['linhas'])} linhas (anotacao oficial do Manga109):\n")

    for linha in gt["linhas"]:
        x1, y1, x2, y2 = linha["bbox"]
        marcas = []
        for char in linha["n1_esperados"]:
            total += 1
            achou = any(_dentro(d.bbox, x1, y1, x2, y2) and d.kanji == char for d in deteccoes)
            if achou:
                hits += 1
            marcas.append(f"{char}{'✓' if achou else '✗'}")
        print(f"  bbox={tuple(linha['bbox'])}  esperado: {' '.join(marcas)}")

    print(f"\n[RESUMO] {hits}/{total} kanji N1 esperados encontrados na regiao certa "
          f"(recall {100*hits/max(1,total):.1f}%)")


if __name__ == "__main__":
    main()
