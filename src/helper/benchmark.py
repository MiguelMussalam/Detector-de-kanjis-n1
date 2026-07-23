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
from collections import Counter

import cv2

from config import ROOT_DIR
from src.pipeline.inference import Pipeline
from src.helper.manga109_align import _dentro, limpar_string
from src.helper.kanjis import get_kanjis

BENCHMARK_DIR = os.path.join(ROOT_DIR, "data", "benchmark")
GT_PATH = os.path.join(BENCHMARK_DIR, "ground_truth.json")

_N1_SET = set(get_kanjis("n1"))


def avaliar_pagina(deteccoes: list, linhas: list) -> dict:
    """
    Compara as deteccoes do pipeline contra as linhas esperadas de UMA pagina.
    Reaproveitado tanto por este benchmark (paginas curadas) quanto por
    `src/helper/corpus_validate.py` (validacao full-corpus) -- mesma logica
    de match, uma so implementacao.

    O "esperado" e' um multiset (contagem real de cada kanji na transcricao
    da linha, via `linha["transcricao"]"), nao um set deduplicado -- um kanji
    que aparece 3x na linha precisa de 3 acertos pra contar recall 100% ali,
    nao 1 (bug real corrigido nesta sessao: contar por set deduplicado
    inflava o recall em linha com caractere repetido, comum em particula/
    pronome).

    Alem de hit/miss, atribui cada miss faltante a detector ou classificador:
    conta quantas deteccoes (de QUALQUER classe) sobram na regiao da linha
    depois de descontar os acertos -- se ainda sobra alguma "candidata" nao
    usada quando uma ocorrencia esperada nao bate, o detector achou uma caixa
    ali e o classificador que errou/rejeitou; se nao sobra nenhuma, o
    detector nunca gerou caixa nenhuma pra aquela ocorrencia. E uma
    heuristica (Manga109 nao da posicao por caractere, so por linha, entao
    nao da pra saber qual caixa "deveria" ser qual ocorrencia), nao uma
    atribuicao exata.
    """
    hits, esperado = 0, 0
    miss_detector, miss_classificador = 0, 0
    detalhe = []
    for linha in linhas:
        x1, y1, x2, y2 = linha["bbox"]
        deteccoes_na_regiao = [d for d in deteccoes if _dentro(d.bbox, x1, y1, x2, y2)]

        esperado_counter = Counter(c for c in limpar_string(linha["transcricao"]) if c in _N1_SET)
        detectado_counter = Counter(d.kanji for d in deteccoes_na_regiao)
        candidatas_sobrando = len(deteccoes_na_regiao)

        marcas = []
        for char, n_esperado in esperado_counter.items():
            esperado += n_esperado
            n_hit = min(n_esperado, detectado_counter.get(char, 0))
            hits += n_hit
            candidatas_sobrando -= n_hit

            n_falta = n_esperado - n_hit
            if n_falta > 0:
                n_class = min(n_falta, max(0, candidatas_sobrando))
                miss_classificador += n_class
                candidatas_sobrando -= n_class
                miss_detector += (n_falta - n_class)
            marcas.append((char, n_hit, n_esperado))
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
            marcas = " ".join(
                f"{c}{n_hit}/{n_esp}{'✓' if n_hit == n_esp else '✗'}"
                for c, n_hit, n_esp in item["marcas"]
            )
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
