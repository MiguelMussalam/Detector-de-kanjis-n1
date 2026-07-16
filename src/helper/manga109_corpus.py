"""
manga109_corpus.py
===================
Constrói ground truth automaticamente para TODO o corpus Manga109 (todos os
volumes/páginas), sem depender do detector -- só usa bbox de linha (<text>)
e transcrição oficial, direto do XML. Mesmo formato de
`data/benchmark/ground_truth.json` (curado manualmente, 4 páginas, baixo
ruído -- continua existindo à parte), mas em escala: aceita uma margem de
erro residual documentada em vez de checagem visual linha a linha (inviável
em ~10.600 páginas).

Filtro heurístico automático (substitui a verificação visual manual):
  1. Script fora do esperado: qualquer caractere fora de
     hiragana/katakana/CJK/pontuação/dígitos/latim marca a linha como
     suspeita -- pega o caso confirmado nesta sessão (letra grega isolada
     em texto de efeito sonoro estilizado corrompido).
  2. Área de bbox implausível pro número de caracteres (ver CORPUS_CELL_AREA_*
     em config.py) -- pega bbox claramente pequena/grande demais pro
     conteúdo declarado.
Nenhum dos dois é garantia de 100%: linhas excluídas ficam registradas (com
motivo) em vez de descartadas silenciosamente, pra permitir auditoria.

Uso:
    python -m src.helper.manga109_corpus [--limit-volumes N] [--volumes v1,v2]
"""

import argparse
import json
import os
import xml.etree.ElementTree as ET
from glob import glob

from tqdm import tqdm

from config import ROOT_DIR, MANGA109_ANNOTATIONS, CORPUS_CELL_AREA_MIN
from src.helper.kanjis import get_kanjis
from src.helper.manga109_align import limpar_string

CORPUS_DIR = os.path.join(ROOT_DIR, "data", "corpus_validation")
GT_PATH = os.path.join(CORPUS_DIR, "ground_truth_full.json")


# Dialogo de manga usa uma variedade enorme de pontuacao/simbolos tipograficos
# legitimos (travessao 〜/―, aspas curvas, coracao, numeral romano, simbolo de
# unidade...) -- tentar montar uma whitelist exaustiva disso se mostrou
# contraproducente (na auditoria inicial, 57/58 "suspeitas" eram pontuacao
# normal, nao corrupcao). Em vez disso, blacklist de blocos Unicode que nunca
# aparecem em japones de verdade -- pega o caso confirmado (letra grega
# isolada em efeito sonoro estilizado corrompido) sem gerar falso positivo
# em cima da tipografia real do corpus.
_BLOCOS_SUSPEITOS = [
    (0x0370, 0x03FF),   # grego e copta
    (0x0400, 0x04FF),   # cirilico
    (0x0530, 0x058F),   # armenio
    (0x0590, 0x05FF),   # hebraico
    (0x0600, 0x06FF),   # arabe
    (0x0900, 0x097F),   # devanagari
    (0x0E00, 0x0E7F),   # tailandes
    (0x10A0, 0x10FF),   # georgiano
    (0x1100, 0x11FF),   # hangul jamo
    (0xAC00, 0xD7A3),   # hangul silabas
]


def _script_suspeito(c: str) -> bool:
    cp = ord(c)
    return any(lo <= cp <= hi for lo, hi in _BLOCOS_SUSPEITOS)


def avaliar_transcricao_confiavel(texto: str, bbox) -> tuple:
    if not texto:
        return False, "transcricao_vazia"

    for c in texto:
        if _script_suspeito(c):
            return False, "script_anomalo"

    # So o piso -- teto de area foi tentado e descartado na auditoria (ver
    # config.py): onomatopeia/titulo dramatico em fonte gigante e convencao
    # normal de mangá pra frase curta, nao sinal de transcricao com caractere
    # faltando. O piso continua util pro caso oposto (bbox estreita demais
    # pro numero de caracteres declarado).
    x1, y1, x2, y2 = bbox
    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_por_char = area / len(texto)
    if area_por_char < CORPUS_CELL_AREA_MIN:
        return False, "bbox_pequena_demais"

    return True, None


def iter_paginas(volumes=None):
    """Gera (volume, indice_pagina, page_element) pra todo o Manga109 (ou subset)."""
    if volumes is None:
        volumes = sorted(
            os.path.splitext(os.path.basename(f))[0]
            for f in glob(os.path.join(MANGA109_ANNOTATIONS, "*.xml"))
        )
    for volume in volumes:
        xml_path = os.path.join(MANGA109_ANNOTATIONS, f"{volume}.xml")
        root = ET.parse(xml_path).getroot()
        for page in root.iter("page"):
            yield volume, int(page.get("index")), page


def construir_linha(text_el, n1_set: set):
    texto_limpo = limpar_string(text_el.text)
    if not texto_limpo:
        return None
    bbox = [
        int(text_el.get("xmin")), int(text_el.get("ymin")),
        int(text_el.get("xmax")), int(text_el.get("ymax")),
    ]
    n1_esperados = sorted(set(c for c in texto_limpo if c in n1_set))
    return {
        "bbox": bbox, "transcricao": text_el.text,
        "n1_esperados": n1_esperados, "_texto_limpo": texto_limpo,
    }


def construir_ground_truth(volumes=None) -> dict:
    n1_set = set(get_kanjis("n1"))
    paginas = []
    n_excluidas = 0
    n_paginas = 0

    for volume, idx, page in tqdm(iter_paginas(volumes), desc="paginas"):
        linhas, excluidas = [], []
        for text_el in page.iter("text"):
            linha = construir_linha(text_el, n1_set)
            if linha is None or not linha["n1_esperados"]:
                continue

            confiavel, motivo = avaliar_transcricao_confiavel(linha["_texto_limpo"], linha["bbox"])
            del linha["_texto_limpo"]
            if confiavel:
                linhas.append(linha)
            else:
                excluidas.append({**linha, "motivo": motivo})
                n_excluidas += 1

        n_paginas += 1
        if linhas or excluidas:
            paginas.append({
                "origem": {"volume": volume, "pagina": idx},
                "linhas": linhas,
                "excluidas": excluidas,
            })

    return {
        "metodologia": (
            "bbox e transcricao vem direto da anotacao oficial do Manga109 (<text>), "
            "mesmo principio do data/benchmark/ground_truth.json curado manualmente -- "
            "mas cobrindo todo o corpus (109 volumes) com filtro automatico de "
            "transcricao suspeita em vez de verificacao visual linha a linha "
            "(ver avaliar_transcricao_confiavel). Ruido residual documentado: nao e "
            "garantia de 100% de precisao -- auditar uma amostra do campo 'excluidas' "
            "de cada pagina antes de confiar no recall agregado em escala."
        ),
        "n_paginas_processadas": n_paginas,
        "n_linhas_excluidas": n_excluidas,
        "paginas": paginas,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit-volumes", type=int, default=None)
    parser.add_argument("--volumes", type=str, default=None, help="lista separada por virgula")
    parser.add_argument("--out", type=str, default=GT_PATH)
    args = parser.parse_args()

    volumes = None
    if args.volumes:
        volumes = args.volumes.split(",")
    elif args.limit_volumes:
        todos = sorted(
            os.path.splitext(os.path.basename(f))[0]
            for f in glob(os.path.join(MANGA109_ANNOTATIONS, "*.xml"))
        )
        volumes = todos[:args.limit_volumes]

    gt = construir_ground_truth(volumes)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(gt, f, ensure_ascii=False, indent=2)

    n_linhas = sum(len(p["linhas"]) for p in gt["paginas"])
    print(f"\n[INFO] {gt['n_paginas_processadas']} paginas processadas, "
          f"{n_linhas} linhas validas, {gt['n_linhas_excluidas']} excluidas pelo filtro")
    print(f"[INFO] Salvo em {args.out}")


if __name__ == "__main__":
    main()
