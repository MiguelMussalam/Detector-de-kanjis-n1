import os
import sys
import time
import numpy as np
import cv2
import mss
from ultralytics import YOLO

def main():
    print("=" * 60)
    print("      YOLO Manga OCR - Real-Time Screen Inference      ")
    print("=" * 60)

    # 1. Definir caminhos e carregar o modelo
    weights_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")
    best_weights = os.path.join(weights_dir, "best.pt")
    fallback_weights = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yolo26n.pt")

    if os.path.exists(best_weights):
        model_path = best_weights
        print(f"[INFO] Carregando modelo treinado: {model_path}")
    elif os.path.exists(fallback_weights):
        model_path = fallback_weights
        print(f"[WARNING] Pesos '{best_weights}' nao encontrados.")
        print(f"[INFO] Usando modelo fallback padrão: {model_path}")
    else:
        print("[ERROR] Nenhum modelo YOLO encontrado!")
        print(f"Por favor, coloque o arquivo de pesos 'best.pt' em: {best_weights}")
        print("Ou certifique-se de ter o modelo base 'yolo26n.pt' na raiz do projeto.")
        sys.exit(1)

    try:
        model = YOLO(model_path)
        print("[INFO] Modelo carregado com sucesso!")
    except Exception as e:
        print(f"[ERROR] Falha ao carregar o modelo YOLO: {e}")
        sys.exit(1)

    # 2. Configurar captura de tela com mss
    with mss.MSS() as sct:
        # Usar monitor primario
        monitors = sct.monitors
        if len(monitors) > 1:
            monitor = monitors[1]
        else:
            monitor = monitors[0]

        print(f"[INFO] Monitor detectado: {monitor['width']}x{monitor['height']} em ({monitor['left']}, {monitor['top']})")

        # Configurar regiao inicial de captura de 800x800 centralizada
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
            # Garante limites minimos e maximos da regiao de captura
            r["width"] = max(100, min(r["width"], monitor["width"]))
            r["height"] = max(100, min(r["height"], monitor["height"]))
            r["left"] = max(monitor["left"], min(r["left"], monitor["left"] + monitor["width"] - r["width"]))
            r["top"] = max(monitor["top"], min(r["top"], monitor["top"] + monitor["height"] - r["height"]))

        clip_region(region)

        print("\nControles do Visualizador (Foque na janela do OpenCV):")
        print("  - [W, A, S, D] : Mover a regiao de captura (Cima, Esquerda, Baixo, Direita)")
        print("  - [R]          : Aumentar tamanho da janela de captura (+50px)")
        print("  - [F]          : Diminuir tamanho da janela de captura (-50px)")
        print("  - [H]          : Imprimir informacoes de depuracao no terminal")
        print("  - [Q]          : Sair do visualizador")
        print("-" * 60)

        prev_time = time.time()
        
        while True:
            start_frame_time = time.time()

            # Capturar regiao da tela
            try:
                screenshot = sct.grab(region)
            except Exception as e:
                print(f"[ERROR] Erro na captura de tela: {e}")
                break

            # Converter BGRA para BGR
            img = np.array(screenshot)
            frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

            # Executar inferencia com YOLO
            # verbose=False evita poluir o terminal a cada frame
            results = model(frame, verbose=False)

            # Renderizar bboxes na imagem
            annotated_frame = results[0].plot()

            # Calcular FPS
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0.0
            prev_time = curr_time

            # Desenhar informações na tela
            cv2.putText(
                annotated_frame, 
                f"FPS: {fps:.1f} | Regiao: {region['width']}x{region['height']}", 
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

            # Mostrar imagem
            cv2.imshow('YOLO Manga OCR - Real Time', annotated_frame)

            # Tratar entrada do usuario
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
                # Centraliza a expansao
                region["left"] -= 25
                region["top"] -= 25
                clip_region(region)
            elif key == ord('f'):
                region["width"] -= 50
                region["height"] -= 50
                # Centraliza a reducao
                region["left"] += 25
                region["top"] += 25
                clip_region(region)
            elif key == ord('h'):
                print(f"[DEBUG] Regiao de Captura: Top={region['top']}, Left={region['left']}, Lg={region['width']}, Al={region['height']}")

        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
