import argparse
import os
import sys
import time
import glob

import numpy as np
import cv2
import mss
from PIL import Image, ImageDraw, ImageFont

from src.pipeline.inference import Pipeline, Deteccao, load_detector
from config import (
    FONTS_DIR, PIPELINE_CLS_CONF_LOW, CLF_OUTROS_LABEL,
    DETECTOR_WEIGHTS_PATH, PIPELINE_DET_CONF, PIPELINE_DET_IOU,
    PIPELINE_DET_MAX_DET, PIPELINE_MIN_BBOX_HEIGHT,
)


def carregar_fonte_overlay(tamanho=18):
    """cv2.putText nao suporta CJK, entao o overlay do kanji previsto usa PIL + fonte japonesa."""
    candidatos = glob.glob(os.path.join(FONTS_DIR, "*.ttf"))
    if not candidatos:
        raise FileNotFoundError(
            "Nenhuma fonte encontrada em assets/fonts/. Rode `python -m src.helper.fonts` primeiro."
        )
    return ImageFont.truetype(candidatos[0], tamanho)


def detectar_somente(detector, frame_bgr,
                     conf_det: float = PIPELINE_DET_CONF,
                     iou_det: float = PIPELINE_DET_IOU) -> list:
    """
    Roda só o detector (sem classificador) -- usado no modo --detector-only pra
    isolar visualmente a cobertura do detector (onde ele acha/nao acha caixa),
    sem o classificador misturado no julgamento visual.
    """
    results = detector(
        frame_bgr, conf=conf_det, iou=iou_det,
        agnostic_nms=True, max_det=PIPELINE_DET_MAX_DET, verbose=False,
    )[0]

    deteccoes = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        if (y2 - y1) < PIPELINE_MIN_BBOX_HEIGHT:
            continue
        deteccoes.append(Deteccao(
            bbox=(int(x1), int(y1), int(x2), int(y2)),
            kanji="", codepoint="",
            confianca_det=float(box.conf[0]), confianca_cls=0.0,
        ))
    return deteccoes


def desenhar_deteccoes_detector(frame_bgr, deteccoes):
    """So desenha bbox + confianca do detector -- sem kanji, nao ha classificador nesse modo."""
    frame = frame_bgr.copy()
    for det in deteccoes:
        x1, y1, x2, y2 = det.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(frame, f"{det.confianca_det:.2f}", (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    return frame


def desenhar_deteccoes_ocr(frame_bgr, resultados, fonte):
    """
    Desenha o quadrilatero + texto bruto que o EasyOCR leu em cada regiao.
    `resultados` vem de reader.readtext(..., detail=1): lista de (bbox_poly, texto, conf).
    Sem filtro N1 aqui -- e' so visualizacao ao vivo pra comparar lado a lado com o
    pipeline proprio; a comparacao formal (com metrica) e' feita por
    src/helper/ocr_baseline_compare.py.
    """
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(img_pil)

    for bbox_poly, texto, conf in resultados:
        pontos = [tuple(int(v) for v in p) for p in bbox_poly]
        cor = (0, 255, 0) if conf >= 0.3 else (255, 140, 0)
        draw.line(pontos + [pontos[0]], fill=cor, width=2)
        x1, y1 = pontos[0]
        draw.text((x1, max(0, y1 - 20)), f"{texto} ({conf:.2f})", font=fonte, fill=cor)

    frame_rgb_out = np.array(img_pil)
    return cv2.cvtColor(frame_rgb_out, cv2.COLOR_RGB2BGR)


def desenhar_deteccoes(frame_bgr, deteccoes, fonte, mostrar_outros=False):
    """
    Desenha bbox + kanji previsto + confianca do classificador em cada deteccao.
    Se mostrar_outros=False (padrao), deteccoes rejeitadas como CLF_OUTROS_LABEL
    nao sao desenhadas — so aparecem caixas de predicoes que podem ser kanji N1.
    Verde: confianca do classificador >= PIPELINE_CLS_CONF_LOW. Laranja: abaixo disso
    (previsao incerta, mas ainda dentro das classes N1).
    """
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(img_pil)

    for det in deteccoes:
        if det.kanji == CLF_OUTROS_LABEL:
            if not mostrar_outros:
                continue
            cor = (140, 140, 140)
        elif det.confianca_cls >= PIPELINE_CLS_CONF_LOW:
            cor = (0, 255, 0)
        else:
            cor = (255, 140, 0)

        x1, y1, x2, y2 = det.bbox

        draw.rectangle([x1, y1, x2, y2], outline=cor, width=2)
        texto = f"{det.kanji} {det.confianca_cls:.2f}"
        draw.text((x1, max(0, y1 - 20)), texto, font=fonte, fill=cor)

    frame_rgb_out = np.array(img_pil)
    return cv2.cvtColor(frame_rgb_out, cv2.COLOR_RGB2BGR)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector-only", action="store_true",
                        help="Roda so o detector (sem classificador) -- mostra so bbox + confianca de deteccao, "
                             "util pra isolar visualmente a cobertura do detector.")
    parser.add_argument("--ocr-baseline", action="store_true",
                        help="Roda o EasyOCR (baseline tradicional) na regiao capturada, pra comparar "
                             "visualmente com o pipeline proprio. Sob demanda (tecla ESPACO), nao a cada "
                             "frame -- EasyOCR na CPU e' lento demais pra isso.")
    args = parser.parse_args()

    modo_detector_only = args.detector_only
    modo_ocr_baseline = args.ocr_baseline
    if modo_detector_only and modo_ocr_baseline:
        print("[ERROR] Use --detector-only OU --ocr-baseline, nao os dois.")
        sys.exit(1)

    if modo_ocr_baseline:
        titulo_modo = "EasyOCR (baseline tradicional)"
    elif modo_detector_only:
        titulo_modo = "Detector (sem classificador)"
    else:
        titulo_modo = "Pipeline Completo (Detector + Classificador N1)"

    print("=" * 60)
    print(f"   {titulo_modo} - Real-Time   ")
    print("=" * 60)

    window_name = "Pipeline Kanji N1 - Real Time"

    fonte = None
    detector = None
    pipeline = None
    ocr_reader = None

    try:
        if modo_ocr_baseline:
            print("[INFO] Carregando EasyOCR (japones)...")
            import easyocr
            ocr_reader = easyocr.Reader(["ja"], gpu=False)
            fonte = carregar_fonte_overlay()
            print("[INFO] EasyOCR carregado com sucesso!")
            print("[AVISO] EasyOCR e' lento na CPU -- por isso aqui ele NAO roda a cada frame. "
                  "Posicione a regiao com [WASD]/[R]/[F] e pressione [ESPACO] pra rodar o OCR sob demanda.")
        elif modo_detector_only:
            print("[INFO] Carregando detector...")
            detector = load_detector(DETECTOR_WEIGHTS_PATH)
            print("[INFO] Detector carregado com sucesso!")
        else:
            print("[INFO] Carregando pipeline (detector + classificador)...")
            pipeline = Pipeline()
            fonte = carregar_fonte_overlay()
            print("[INFO] Pipeline carregado com sucesso!")
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    with mss.MSS() as sct:
        monitors = sct.monitors
        if len(monitors) > 1:
            monitor = monitors[1]
        else:
            monitor = monitors[0]

        print(f"[INFO] Monitor detectado: {monitor['width']}x{monitor['height']} em ({monitor['left']}, {monitor['top']})")

        width, height = 1920, 1080
        width = min(width, monitor["width"])
        height = min(height, monitor["height"])

        left = monitor["left"] + (monitor["width"] - width) // 2
        top = monitor["top"] + (monitor["height"] - height) // 2

        region = {
            "top": top,
            "left": left,
            "width": width,
            "height": height
        }

        def clip_region(r):
            r["width"] = max(100, min(r["width"], monitor["width"]))
            r["height"] = max(100, min(r["height"], monitor["height"]))
            r["left"] = max(monitor["left"], min(r["left"], monitor["left"] + monitor["width"] - r["width"]))
            r["top"] = max(monitor["top"], min(r["top"], monitor["top"] + monitor["height"] - r["height"]))

        clip_region(region)

        print("\nControles do Visualizador (Foque na janela do OpenCV):")
        print("  - [W, A, S, D] : Mover a regiao de captura (Cima, Esquerda, Baixo, Direita)")
        print("  - [R]          : Aumentar tamanho da janela de captura (+50px)")
        print("  - [F]          : Diminuir tamanho da janela de captura (-50px)")
        if not modo_detector_only and not modo_ocr_baseline:
            print("  - [O]          : Mostrar/esconder deteccoes classificadas como OUTROS")
        if modo_ocr_baseline:
            print("  - [ESPACO]     : Rodar o EasyOCR no frame atual (e' sob demanda, nao continuo -- "
                  "CPU e' lenta demais pra rodar a cada frame)")
        print("  - [H]          : Imprimir informacoes de depuracao no terminal")
        print("  - [Q]          : Sair do visualizador")
        print("-" * 60)

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        prev_time = time.time()
        mostrar_outros = False
        if not modo_detector_only and not modo_ocr_baseline:
            print(f"[INFO] Exibindo OUTROS: {mostrar_outros} (pressione [O] para alternar)")
        deteccoes = []

        while True:
            try:
                screenshot = sct.grab(region)
            except Exception as e:
                print(f"[ERROR] Erro na captura de tela: {e}")
                break

            img = np.array(screenshot)
            frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

            if modo_ocr_baseline:
                # Sob demanda (tecla [ESPACO]) -- rodar a cada frame trava a navegacao,
                # ja que o EasyOCR na CPU leva segundos por chamada.
                annotated_frame = desenhar_deteccoes_ocr(frame, deteccoes, fonte)
            elif modo_detector_only:
                deteccoes = detectar_somente(detector, frame)
                annotated_frame = desenhar_deteccoes_detector(frame, deteccoes)
            else:
                deteccoes = pipeline.predict(frame)
                annotated_frame = desenhar_deteccoes(frame, deteccoes, fonte, mostrar_outros)

            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0.0
            prev_time = curr_time

            cv2.putText(
                annotated_frame,
                f"[{titulo_modo}] FPS: {fps:.1f} | Regiao: {region['width']}x{region['height']} | Deteccoes: {len(deteccoes)}",
                (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )
            rodape = ("[Q] Sair | [WASD] Mover | [R/F] Redimensionar | [ESPACO] Rodar OCR"
                      if modo_ocr_baseline else
                      "[Q] Sair | [WASD] Mover | [R/F] Redimensionar")
            cv2.putText(
                annotated_frame,
                rodape,
                (15, annotated_frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

            cv2.imshow(window_name, annotated_frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("[INFO] Fechando visualizador...")
                break
            elif key == ord('w'):
                region["top"] -= 20
                clip_region(region)
                deteccoes = [] if modo_ocr_baseline else deteccoes
            elif key == ord('s'):
                region["top"] += 20
                clip_region(region)
                deteccoes = [] if modo_ocr_baseline else deteccoes
            elif key == ord('a'):
                region["left"] -= 20
                clip_region(region)
                deteccoes = [] if modo_ocr_baseline else deteccoes
            elif key == ord('d'):
                region["left"] += 20
                clip_region(region)
                deteccoes = [] if modo_ocr_baseline else deteccoes
            elif key == ord('r'):
                region["width"] += 50
                region["height"] += 50
                region["left"] -= 25
                region["top"] -= 25
                clip_region(region)
                deteccoes = [] if modo_ocr_baseline else deteccoes
            elif key == ord('f'):
                region["width"] -= 50
                region["height"] -= 50
                region["left"] += 25
                region["top"] += 25
                clip_region(region)
                deteccoes = [] if modo_ocr_baseline else deteccoes
            elif key == ord(' ') and modo_ocr_baseline:
                print("[INFO] Rodando EasyOCR no frame atual...")
                t0 = time.time()
                frame_rgb_in = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                deteccoes = ocr_reader.readtext(frame_rgb_in, detail=1, paragraph=False)
                print(f"[INFO] OCR concluido em {time.time() - t0:.2f}s -- {len(deteccoes)} regiao(oes) de texto")
            elif key == ord('o') and not modo_detector_only and not modo_ocr_baseline:
                mostrar_outros = not mostrar_outros
                print(f"[INFO] Exibindo OUTROS: {mostrar_outros}")
            elif key == ord('h'):
                print(f"[DEBUG] Regiao de Captura: Top={region['top']}, Left={region['left']}, Lg={region['width']}, Al={region['height']}")
                print(f"[DEBUG] {len(deteccoes)} deteccoes no frame atual:")
                if modo_ocr_baseline:
                    for bbox_poly, texto, conf in deteccoes:
                        print(f"    texto={texto!r} conf={conf:.2f} bbox={bbox_poly}")
                elif modo_detector_only:
                    for det in deteccoes:
                        print(f"    bbox={det.bbox} det={det.confianca_det:.2f}")
                else:
                    for det in deteccoes:
                        print(f"    {det.kanji} ({det.codepoint}) det={det.confianca_det:.2f} cls={det.confianca_cls:.2f}")

        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
