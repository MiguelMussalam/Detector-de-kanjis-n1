import os
import sys
import time
import glob

import numpy as np
import cv2
import mss
from PIL import Image, ImageDraw, ImageFont

from src.pipeline.inference import Pipeline
from config import FONTS_DIR, PIPELINE_CLS_CONF_LOW, CLF_OUTROS_LABEL


def carregar_fonte_overlay(tamanho=18):
    """cv2.putText nao suporta CJK, entao o overlay do kanji previsto usa PIL + fonte japonesa."""
    candidatos = glob.glob(os.path.join(FONTS_DIR, "*.ttf"))
    if not candidatos:
        raise FileNotFoundError(
            "Nenhuma fonte encontrada em assets/fonts/. Rode `python -m src.helper.fonts` primeiro."
        )
    return ImageFont.truetype(candidatos[0], tamanho)


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
    print("=" * 60)
    print("   Pipeline Completo (Detector + Classificador N1) - Real-Time   ")
    print("=" * 60)

    print("[INFO] Carregando pipeline (detector + classificador)...")
    try:
        pipeline = Pipeline()
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
    fonte = carregar_fonte_overlay()
    print("[INFO] Pipeline carregado com sucesso!")

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
        print("  - [O]          : Mostrar/esconder deteccoes classificadas como OUTROS")
        print("  - [H]          : Imprimir informacoes de depuracao no terminal")
        print("  - [Q]          : Sair do visualizador")
        print("-" * 60)

        prev_time = time.time()
        mostrar_outros = False
        print(f"[INFO] Exibindo OUTROS: {mostrar_outros} (pressione [O] para alternar)")

        while True:
            try:
                screenshot = sct.grab(region)
            except Exception as e:
                print(f"[ERROR] Erro na captura de tela: {e}")
                break

            img = np.array(screenshot)
            frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

            deteccoes = pipeline.predict(frame)
            annotated_frame = desenhar_deteccoes(frame, deteccoes, fonte, mostrar_outros)

            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0.0
            prev_time = curr_time

            cv2.putText(
                annotated_frame,
                f"FPS: {fps:.1f} | Regiao: {region['width']}x{region['height']} | Deteccoes: {len(deteccoes)}",
                (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )
            cv2.putText(
                annotated_frame,
                "[Q] Sair | [WASD] Mover | [R/F] Redimensionar",
                (15, annotated_frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

            cv2.imshow('Pipeline Kanji N1 - Real Time', annotated_frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("[INFO] Fechando visualizador...")
                break
            elif key == ord('w'):
                region["top"] -= 20
                clip_region(region)
            elif key == ord('s'):
                region["top"] += 20
                clip_region(region)
            elif key == ord('a'):
                region["left"] -= 20
                clip_region(region)
            elif key == ord('d'):
                region["left"] += 20
                clip_region(region)
            elif key == ord('r'):
                region["width"] += 50
                region["height"] += 50
                region["left"] -= 25
                region["top"] -= 25
                clip_region(region)
            elif key == ord('f'):
                region["width"] -= 50
                region["height"] -= 50
                region["left"] += 25
                region["top"] += 25
                clip_region(region)
            elif key == ord('o'):
                mostrar_outros = not mostrar_outros
                print(f"[INFO] Exibindo OUTROS: {mostrar_outros}")
            elif key == ord('h'):
                print(f"[DEBUG] Regiao de Captura: Top={region['top']}, Left={region['left']}, Lg={region['width']}, Al={region['height']}")
                print(f"[DEBUG] {len(deteccoes)} deteccoes no frame atual:")
                for det in deteccoes:
                    print(f"    {det.kanji} ({det.codepoint}) det={det.confianca_det:.2f} cls={det.confianca_cls:.2f}")

        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
