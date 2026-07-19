"""
benchmark.py
============
Roda o pipeline completo (detector + classificador) em cada página de
referência fixa (data/benchmark/*.jpg + ground_truth.json) e compara contra
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
GT_PATH = os.path.join(BENCHMARK_DIR, "ground_truth.json")


def avaliar_pagina(deteccoes: list, linhas: list) -> dict:
    """
    Compara as deteccoes do pipeline contra as linhas esperadas de UMA pagina.
    Reaproveitado tanto por este benchmark (paginas curadas) quanto por
    `src/helper/corpus_validate.py` (validacao full-corpus) -- mesma logica
    de match, uma so implementacao.

    Alem de hit/miss, atribui cada miss a detector ou classificador: conta
    quantas deteccoes (de QUALQUER classe) caem na regiao da linha -- se ainda
    sobra alguma "candidata" nao usada quando um kanji esperado nao bate,
    o detector achou uma caixa ali e o classificador que errou/rejeitou;
    se nao sobra nenhuma, o detector nunca gerou caixa nenhuma pra aquele
    caractere. E uma heuristica (Manga109 nao da posicao por caractere, so
    por linha, entao nao da pra saber qual caixa "deveria" ser qual char),
    nao uma atribuicao exata.
    """
    hits, esperado = 0, 0
    miss_detector, miss_classificador = 0, 0
    detalhe = []
    for linha in linhas:
        x1, y1, x2, y2 = linha["bbox"]
        deteccoes_na_regiao = [d for d in deteccoes if _dentro(d.bbox, x1, y1, x2, y2)]
        candidatas_sobrando = len(deteccoes_na_regiao)

        marcas = []
        for char in linha["n1_esperados"]:
            esperado += 1
            achou = any(d.kanji == char for d in deteccoes_na_regiao)
            if achou:
                hits += 1
                candidatas_sobrando -= 1
                marcas.append((char, True, None))
            elif candidatas_sobrando > 0:
                miss_classificador += 1
                candidatas_sobrando -= 1
                marcas.append((char, False, "classificador"))
            else:
                miss_detector += 1
                marcas.append((char, False, "detector"))
        detalhe.append({"bbox": linha["bbox"], "marcas": marcas})
    return {
        "hits": hits, "esperado": esperado,
        "miss_detector": miss_detector, "miss_classificador": miss_classificador,
        "detalhe": detalhe,
    }


def main():
    with open(GT_PATH, encoding="utf-8") as f:
        gt = json.load(f)

    print("[INFO] Carregando pipeline...")
    pipeline = Pipeline()

    hits_total, esperado_total = 0, 0
    deteccoes_total, outros_total = 0, 0

    for pagina in gt["paginas"]:
        img_path = os.path.join(BENCHMARK_DIR, pagina["imagem"])
        frame = cv2.imread(img_path)
        if frame is None:
            raise FileNotFoundError(f"Nao foi possivel ler {img_path}")

        origem = pagina["origem"]
        print(f"\n[INFO] Rodando em {pagina['imagem']} ({origem['volume']} pag {origem['pagina']})...")
        deteccoes = pipeline.predict(frame)

        n_outros = sum(1 for d in deteccoes if d.codepoint == "OUTROS")
        deteccoes_total += len(deteccoes)
        outros_total += n_outros
        print(f"  [CONTEXTO] {len(deteccoes)} deteccoes, {n_outros} "
              f"({100*n_outros/max(1,len(deteccoes)):.1f}%) classificadas como OUTROS")

        resultado = avaliar_pagina(deteccoes, pagina["linhas"])
        for item in resultado["detalhe"]:
            marcas = " ".join(f"{c}{'✓' if ok else '✗'}" for c, ok in item["marcas"])
            print(f"    bbox={tuple(item['bbox'])}  esperado: {marcas}")

        print(f"  [PAGINA] {resultado['hits']}/{resultado['esperado']} encontrados "
              f"(recall {100*resultado['hits']/max(1,resultado['esperado']):.1f}%)")
        hits_total += resultado["hits"]
        esperado_total += resultado["esperado"]

    print(f"\n[RESUMO GERAL] {deteccoes_total} deteccoes em {len(gt['paginas'])} paginas, "
          f"{outros_total} ({100*outros_total/max(1,deteccoes_total):.1f}%) classificadas como OUTROS")
    print(f"[RESUMO GERAL] {hits_total}/{esperado_total} kanji N1 esperados encontrados na regiao certa "
          f"(recall {100*hits_total/max(1,esperado_total):.1f}%)")


if __name__ == "__main__":
    main()
