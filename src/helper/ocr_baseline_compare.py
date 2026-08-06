"""
ocr_baseline_compare.py
========================
Compara o recall do nosso pipeline (detector + classificador) contra uma
ferramenta de OCR tradicional (EasyOCR, com suporte a japonês) na MESMA
tarefa: reconhecer kanji N1 em linhas de diálogo reais do Manga109.

Testa a Hipótese 1 da proposta formal da IC ("YOLOv8 terá mAP/recall
superior a ferramentas de OCR tradicionais na detecção de Kanjis N1").

Três motores (`--engine`): `easyocr` (padrão), `tesseract` (segundo baseline,
reforça a conclusão sem depender de uma única ferramenta -- ver EXPERIMENTS.md)
e `mangaocr` (kha-white/manga-ocr-base, especializado em texto de mangá --
não é um dos baselines que a Hipótese 1 compara, testa se a conclusão também
vale contra uma ferramenta já especializada no domínio, não só genéricas).
Tesseract precisa estar instalado no sistema (não é pip) com `jpn_vert.traineddata`
disponível (ver `TESSERACT_CMD`/`TESSDATA_DIR` em config.py).

Dois modos (`--modo`):

  - `recorte` (padrão): isola o reconhecimento da detecção de propósito --
    recorta exatamente a MESMA linha do nosso ground truth (bbox oficial
    `<text>` do Manga109) e roda o OCR só nesse recorte. A comparação fica
    restrita a "dado que sabemos onde o texto está, quem lê melhor o kanji
    raro" -- onde nossa contribuição de fato mira (classificação, não
    detecção de texto em geral).
  - `pagina`: roda o EasyOCR na PÁGINA INTEIRA, sem dar a bbox de graça --
    ele acha o texto sozinho (detecção + reconhecimento próprios). Fecha a
    lacuna da Hipótese 1 como está literalmente escrita na proposta formal
    ("recall de DETECÇÃO", não só leitura numa região já apontada). Métrica
    é por página (multiset agregado de todas as linhas daquela página, não
    dá pra saber qual caractere veio de qual linha sem alinhamento -- e não
    é o que importa aqui). Mais lento por página (imagem bem maior que um
    recorte de linha); sem CER nem grade visual nesse modo por ora (escopo
    reduzido de propósito, ver EXPERIMENTS.md).

Métrica: mesma lógica de multiset já usada em `benchmark.py`/`corpus_validate.py`
-- conta cada ocorrência de kanji N1 na transcrição esperada (via Counter),
compara contra quantas vezes cada kanji N1 aparece na saída do OCR pra
aquele recorte (também via Counter, após filtrar pelo conjunto N1), usando
min(esperado, encontrado) por classe.

Também calcula CER (character error rate, via distância de edição) sobre a
transcrição LIMPA completa (não só kanji N1) -- serve pra separar "o EasyOCR
lê mal kanji N1 especificamente" de "o EasyOCR lê mal texto de mangá em
geral" (kana, pontuação, tudo incluso). O detalhe linha-a-linha (transcrição,
o que o OCR leu, CER individual) é salvo em CSV pra permitir essa análise
sem precisar rodar o EasyOCR de novo.

Uso:
    python -m src.helper.ocr_baseline_compare --limit-volumes 3
    python -m src.helper.ocr_baseline_compare --volumes LancelotFullThrottle
"""

import argparse
import csv
import json
import os
from collections import Counter

import cv2
from tqdm import tqdm

from config import ROOT_DIR, MANGA109_IMAGES, VISUAL_COMPARACAO_DIR
from src.helper.kanjis import get_kanjis
from src.helper.manga109_align import limpar_string
from src.helper.manga109_corpus import GT_PATH as GT_PATH_DEFAULT
from src.helper.corpus_validate import _filtrar_paginas

RESULT_PATH = os.path.join(ROOT_DIR, "data", "corpus_validation", "resultado_ocr_baseline.json")
DETALHE_PATH = os.path.join(ROOT_DIR, "data", "corpus_validation", "ocr_baseline_detalhe.csv")

_N1_SET = set(get_kanjis("n1"))


def contar_n1(texto: str) -> Counter:
    """Conta ocorrências de kanji N1 numa string (filtra pontuação e não-N1)."""
    return Counter(c for c in limpar_string(texto) if c in _N1_SET)


def distancia_edicao(a: str, b: str) -> int:
    """Distância de Levenshtein simples (insercao/remocao/substituicao, custo 1)."""
    if not a:
        return len(b)
    if not b:
        return len(a)
    anterior = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        atual = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            custo = 0 if ca == cb else 1
            atual[j] = min(anterior[j] + 1, atual[j - 1] + 1, anterior[j - 1] + custo)
        anterior = atual
    return anterior[-1]


def _construir_leitor(engine: str):
    """
    Retorna uma função `imagem_gray -> texto_ocr`, escondendo a diferença
    entre motores -- o resto do script (scoring, CER, grade visual) não
    precisa saber qual OCR está por trás.
    """
    if engine == "easyocr":
        print("[INFO] Carregando EasyOCR (japones)...")
        import easyocr
        reader = easyocr.Reader(["ja"], gpu=False)
        return lambda img: "".join(reader.readtext(img, detail=0, paragraph=False))

    if engine == "tesseract":
        import os as _os
        import pytesseract
        from config import TESSERACT_CMD, TESSDATA_DIR

        # TESSDATA_PREFIX via variavel de ambiente, nao via --tessdata-dir no
        # config string -- o caminho do projeto tem espaco ("IC - OCR"), e o
        # jeito que pytesseract monta a linha de comando quebra com aspas
        # embutidas nesse caso.
        _os.environ["TESSDATA_PREFIX"] = TESSDATA_DIR
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
        print(f"[INFO] Usando Tesseract ({TESSERACT_CMD}, tessdata={TESSDATA_DIR})")
        return lambda img: pytesseract.image_to_string(img, lang="jpn_vert")

    if engine == "mangaocr":
        # Especializado em texto de manga (kha-white/manga-ocr-base) --
        # diferente do EasyOCR/Tesseract (generalistas), nao e' um dos
        # baselines que a Hipotese 1 da proposta compara -- serve pra
        # verificar se a conclusao (OCR tradicional perde pro pipeline
        # proprio) tambem vale contra uma ferramenta ja especializada no
        # dominio, nao so contra ferramentas genericas.
        print("[INFO] Carregando manga-ocr...")
        from manga_ocr import MangaOcr
        from PIL import Image
        mocr = MangaOcr()
        return lambda img: mocr(Image.fromarray(img).convert("RGB"))

    raise ValueError(f"Motor de OCR desconhecido: {engine}")


def avaliar_linha_ocr(ler_fn, frame_gray, bbox):
    """
    Roda o OCR no recorte da linha (mesma bbox do nosso ground truth).
    Retorna (encontrado_counter, texto_ocr_bruto) -- o texto bruto e' guardado
    pra auditoria visual (ver salvar_grade_visual em src/helper/grade_visual.py),
    nao so pro calculo de hits.
    """
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    recorte = frame_gray[max(0, y1):y2, max(0, x1):x2]
    if recorte.size == 0:
        return Counter(), ""

    texto_ocr = ler_fn(recorte)
    return contar_n1(texto_ocr), texto_ocr


def avaliar_pagina_ocr(ler_fn, frame_gray, linhas):
    """
    Roda o OCR na página inteira (ver docstring do módulo, modo `pagina`).
    Agrega o esperado de todas as linhas -- retorna (encontrado_counter,
    esperado_counter, texto_ocr_bruto).
    """
    esperado_counter = Counter()
    for linha in linhas:
        esperado_counter.update(contar_n1(linha["transcricao"]))
    if not esperado_counter:
        return Counter(), esperado_counter, ""

    texto_ocr = ler_fn(frame_gray)
    return contar_n1(texto_ocr), esperado_counter, texto_ocr


def _rodar_modo_recorte(ler_fn, paginas, args):
    hits_total, esperado_total = 0, 0
    cer_edicoes_total, cer_chars_total = 0, 0
    por_volume = {}
    amostras_auditoria = []
    detalhe_linhas = []

    for pagina in tqdm(paginas, desc="paginas"):
        origem = pagina["origem"]
        img_path = os.path.join(MANGA109_IMAGES, origem["volume"], f"{origem['pagina']:03d}.jpg")
        frame = cv2.imread(img_path)
        if frame is None:
            continue
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        for linha in pagina["linhas"]:
            esperado_counter = contar_n1(linha["transcricao"])
            if not esperado_counter:
                continue
            encontrado_counter, texto_ocr = avaliar_linha_ocr(ler_fn, frame_gray, linha["bbox"])

            linha_hits = sum(min(n, encontrado_counter.get(c, 0)) for c, n in esperado_counter.items())
            linha_esperado = sum(esperado_counter.values())
            hits_total += linha_hits
            esperado_total += linha_esperado

            transcricao_limpa = limpar_string(linha["transcricao"])
            ocr_limpo = limpar_string(texto_ocr)
            edicoes = distancia_edicao(transcricao_limpa, ocr_limpo)
            tam_esperado_texto = len(transcricao_limpa)
            cer_edicoes_total += edicoes
            cer_chars_total += tam_esperado_texto
            cer_linha = 100 * edicoes / max(1, tam_esperado_texto)

            vol_stats = por_volume.setdefault(
                origem["volume"],
                {"hits": 0, "esperado": 0, "cer_edicoes": 0, "cer_chars": 0},
            )
            vol_stats["hits"] += linha_hits
            vol_stats["esperado"] += linha_esperado
            vol_stats["cer_edicoes"] += edicoes
            vol_stats["cer_chars"] += tam_esperado_texto

            detalhe_linhas.append({
                "volume": origem["volume"],
                "pagina": origem["pagina"],
                "transcricao": linha["transcricao"],
                "texto_ocr": texto_ocr,
                "hits_n1": linha_hits,
                "esperado_n1": linha_esperado,
                "distancia_edicao": edicoes,
                "tam_transcricao_limpa": tam_esperado_texto,
                "cer_pct": round(cer_linha, 1),
            })

            if args.visual_amostra > 0:
                x1, y1, x2, y2 = [int(round(v)) for v in linha["bbox"]]
                recorte_cor = frame[max(0, y1):y2, max(0, x1):x2]
                if recorte_cor.size > 0:
                    amostras_auditoria.append({
                        "recorte": recorte_cor.copy(),
                        "transcricao": linha["transcricao"],
                        "texto_ocr": texto_ocr,
                        "hits": linha_hits,
                        "esperado": linha_esperado,
                    })

    cer_global_pct = 100 * cer_edicoes_total / max(1, cer_chars_total)
    resumo = {
        "engine": args.engine,
        "modo": "recorte",
        "n_paginas": len(paginas),
        "hits_total": hits_total,
        "esperado_total": esperado_total,
        "recall_pct": 100 * hits_total / max(1, esperado_total),
        "cer_pct": round(cer_global_pct, 1),
        "cer_edicoes_total": cer_edicoes_total,
        "cer_chars_total": cer_chars_total,
        "por_volume": {
            vol: {
                **s,
                "recall_pct": 100 * s["hits"] / max(1, s["esperado"]),
                "cer_pct": round(100 * s["cer_edicoes"] / max(1, s["cer_chars"]), 1),
            }
            for vol, s in por_volume.items()
        },
    }

    with open(args.out_detalhe, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(detalhe_linhas[0].keys()) if detalhe_linhas else [])
        writer.writeheader()
        writer.writerows(detalhe_linhas)

    print(f"\n[RESUMO] {args.engine} (recorte): recall N1 = {hits_total}/{esperado_total} ({resumo['recall_pct']:.1f}%)")
    print(f"[RESUMO] {args.engine} (recorte): CER (texto completo, nao so N1) = {cer_global_pct:.1f}% "
          f"({cer_edicoes_total} edicoes / {cer_chars_total} caracteres)")
    print(f"[INFO] Detalhe linha-a-linha salvo em {args.out_detalhe}")

    if args.visual_amostra > 0:
        from src.helper.grade_visual import salvar_grade_visual

        itens = [{
            "imagem": cv2.cvtColor(item["recorte"], cv2.COLOR_BGR2RGB),
            "titulo": (f"esperado: {item['transcricao']}\n"
                       f"OCR leu: {item['texto_ocr'] or '(vazio)'}\n"
                       f"hits N1: {item['hits']}/{item['esperado']}"),
            "cor": "green" if (item["hits"] >= item["esperado"] and item["esperado"] > 0) else "red",
        } for item in amostras_auditoria]

        salvar_grade_visual(itens, os.path.join(VISUAL_COMPARACAO_DIR, f"{args.engine}.png"),
                            n=args.visual_amostra, cols=4, figsize_cel=(4.5, 3.2))

    return resumo


def _rodar_modo_pagina(ler_fn, paginas, args):
    print(f"[AVISO] Modo 'pagina' roda deteccao+reconhecimento do {args.engine} na imagem "
          "inteira -- bem mais lento por pagina que o modo 'recorte'. Teste com "
          "--limit-volumes pequeno antes de rodar a amostra inteira.")

    hits_total, esperado_total = 0, 0
    por_volume = {}
    detalhe_paginas = []

    for pagina in tqdm(paginas, desc="paginas"):
        origem = pagina["origem"]
        img_path = os.path.join(MANGA109_IMAGES, origem["volume"], f"{origem['pagina']:03d}.jpg")
        frame = cv2.imread(img_path)
        if frame is None:
            continue
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        encontrado_counter, esperado_counter, texto_ocr = avaliar_pagina_ocr(ler_fn, frame_gray, pagina["linhas"])
        if not esperado_counter:
            continue

        pagina_hits = sum(min(n, encontrado_counter.get(c, 0)) for c, n in esperado_counter.items())
        pagina_esperado = sum(esperado_counter.values())
        hits_total += pagina_hits
        esperado_total += pagina_esperado

        vol_stats = por_volume.setdefault(origem["volume"], {"hits": 0, "esperado": 0})
        vol_stats["hits"] += pagina_hits
        vol_stats["esperado"] += pagina_esperado

        detalhe_paginas.append({
            "volume": origem["volume"],
            "pagina": origem["pagina"],
            "hits_n1": pagina_hits,
            "esperado_n1": pagina_esperado,
            "texto_ocr_tamanho": len(texto_ocr),
        })

    resumo = {
        "engine": args.engine,
        "modo": "pagina",
        "n_paginas": len(paginas),
        "hits_total": hits_total,
        "esperado_total": esperado_total,
        "recall_pct": 100 * hits_total / max(1, esperado_total),
        "por_volume": {
            vol: {**s, "recall_pct": 100 * s["hits"] / max(1, s["esperado"])}
            for vol, s in por_volume.items()
        },
    }

    with open(args.out_detalhe, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(detalhe_paginas[0].keys()) if detalhe_paginas else [])
        writer.writeheader()
        writer.writerows(detalhe_paginas)

    print(f"\n[RESUMO] {args.engine} (pagina inteira): recall N1 = {hits_total}/{esperado_total} "
          f"({resumo['recall_pct']:.1f}%)")
    print(f"[INFO] Detalhe por pagina salvo em {args.out_detalhe}")
    return resumo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground-truth", default=GT_PATH_DEFAULT)
    parser.add_argument("--volumes", type=str, default=None, help="lista separada por virgula")
    parser.add_argument("--limit-volumes", type=int, default=None)
    parser.add_argument("--sample-paginas-por-volume", type=int, default=None)
    parser.add_argument("--engine", choices=["easyocr", "tesseract", "mangaocr"], default="easyocr",
                        help="Motor de OCR. 'tesseract' precisa estar instalado no sistema "
                             "(nao e' pip) com jpn_vert.traineddata disponivel. 'mangaocr' e' "
                             "especializado em texto de manga, nao e' baseline da Hipotese 1.")
    parser.add_argument("--modo", choices=["recorte", "pagina"], default="recorte",
                        help="'recorte' = bbox da linha dada de graca (isola reconhecimento). "
                             "'pagina' = o motor acha o texto sozinho na pagina inteira "
                             "(deteccao+reconhecimento, fecha a Hipotese 1 como esta escrita).")
    parser.add_argument("--out", default=RESULT_PATH)
    parser.add_argument("--out-detalhe", default=DETALHE_PATH,
                        help="CSV com o detalhe linha-a-linha (transcricao, o que o OCR leu, CER individual).")
    parser.add_argument("--visual-amostra", type=int, default=24,
                        help="Quantas linhas amostrar pra grade de auditoria visual "
                             "(recorte real + esperado + o que o OCR leu). Só vale pro modo "
                             "'recorte'. 0 desativa.")
    args = parser.parse_args()

    sufixo = ("" if args.engine == "easyocr" else f"_{args.engine}") + \
             ("" if args.modo == "recorte" else f"_{args.modo}")
    if args.out_detalhe == DETALHE_PATH and sufixo:
        args.out_detalhe = os.path.join(ROOT_DIR, "data", "corpus_validation", f"ocr_baseline_detalhe{sufixo}.csv")
    if args.out == RESULT_PATH and sufixo:
        args.out = os.path.join(ROOT_DIR, "data", "corpus_validation", f"resultado_ocr_baseline{sufixo}.json")

    with open(args.ground_truth, encoding="utf-8") as f:
        gt = json.load(f)

    volumes = args.volumes.split(",") if args.volumes else None
    if args.limit_volumes and not volumes:
        todos = sorted(set(p["origem"]["volume"] for p in gt["paginas"]))
        volumes = todos[:args.limit_volumes]

    paginas = _filtrar_paginas(
        gt, volumes=volumes, sample_por_volume=args.sample_paginas_por_volume
    )
    print(f"[INFO] Avaliando {len(paginas)} paginas (motor={args.engine}, modo={args.modo})...")

    ler_fn = _construir_leitor(args.engine)

    if args.modo == "pagina":
        resumo = _rodar_modo_pagina(ler_fn, paginas, args)
    else:
        resumo = _rodar_modo_recorte(ler_fn, paginas, args)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(resumo, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Resumo salvo em {args.out}")


if __name__ == "__main__":
    main()
