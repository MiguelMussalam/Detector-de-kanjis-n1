import random
import numpy as np
import cv2
import albumentations as A
from PIL import Image

from config import (
    BLUR_PROB, BLUR_SIGMA_MIN, BLUR_SIGMA_MAX,
    MORFO_PROB, MORFO_K_MIN, MORFO_K_MAX, MORFO_ITER_MIN, MORFO_ITER_MAX,
    RUIDO_PROB, RUIDO_STD_MIN, RUIDO_STD_MAX,
    ROTACAO_PROB, ROTACAO_MAX
)

# ---------------------------------------------------------------------------
# API publica
# ---------------------------------------------------------------------------

def aplicar_degradacoes(imagem: Image.Image, bboxes: list = None) -> tuple:
    """
    Aplica o pipeline completo de degradacao 2D a uma imagem PIL.
    Se bboxes forem fornecidas (no formato COCO: x_min, y_min, width, height),
    elas também serão transformadas pelo pipeline geométrico (rotação microscópica).

    Ordem cronológica física das melhorias:
      1. Rotação microscópica: simula desalinhamento de scan.
         Utiliza Albumentations ShiftScaleRotate se houver bboxes para rotacioná-las
         corretamente em conjunto com a imagem. Caso contrário, usa OpenCV diretamente.
      2. Blur: vai de quase imperceptível a bastante desfocado.
      3. Morfologia: erosão XOR dilatação com kernel e iterações variáveis.
      4. Ruído gaussiano: de quase imperceptível a muito ruidoso.

    Retorna uma tupla (nova_imagem_pil, bboxes_transformadas).
    """
    arr = np.array(imagem.convert("RGB"), dtype=np.uint8)
    h_img, w_img = arr.shape[:2]

    # 1 — Rotação microscópica (geometria)
    if bboxes is not None and len(bboxes) > 0:
        valid_bboxes = []
        for x, y, w, h in bboxes:
            x_min = max(0.0, min(float(w_img - 1), float(x)))
            y_min = max(0.0, min(float(h_img - 1), float(y)))
            x_max = max(0.0, min(float(w_img), float(x + w)))
            y_max = max(0.0, min(float(h_img), float(y + h)))
            
            new_w = x_max - x_min
            new_h = y_max - y_min
            
            if new_w > 0 and new_h > 0:
                valid_bboxes.append([x_min, y_min, new_w, new_h])

        if len(valid_bboxes) > 0:
            class_labels = [0] * len(valid_bboxes)
            transform = A.Compose([
                A.ShiftScaleRotate(
                    shift_limit=0.0,
                    scale_limit=0.0,
                    rotate_limit=(-ROTACAO_MAX, ROTACAO_MAX),
                    interpolation=cv2.INTER_LINEAR,
                    border_mode=cv2.BORDER_REPLICATE,
                    p=ROTACAO_PROB
                )
            ], bbox_params=A.BboxParams(format='coco', label_fields=['class_labels']))
            
            transformed = transform(image=arr, bboxes=valid_bboxes, class_labels=class_labels)
            arr = transformed["image"]
            bboxes = [(int(b[0]), int(b[1]), int(b[2]), int(b[3])) for b in transformed["bboxes"]]
        else:
            # Sem bboxes válidas, aplica rotação apenas na imagem
            if random.random() < ROTACAO_PROB:
                angulo = random.uniform(-ROTACAO_MAX, ROTACAO_MAX)
                M = cv2.getRotationMatrix2D((w_img // 2, h_img // 2), angulo, 1.0)
                arr = cv2.warpAffine(arr, M, (w_img, h_img), borderMode=cv2.BORDER_REPLICATE)
            bboxes = []
    else:
        # Se bboxes for None ou vazia, aplica rotação diretamente na imagem
        if random.random() < ROTACAO_PROB:
            angulo = random.uniform(-ROTACAO_MAX, ROTACAO_MAX)
            M = cv2.getRotationMatrix2D((w_img // 2, h_img // 2), angulo, 1.0)
            arr = cv2.warpAffine(arr, M, (w_img, h_img), borderMode=cv2.BORDER_REPLICATE)

    # 2 — Blur: vai de quase imperceptível a bastante desfocado
    if random.random() < BLUR_PROB:
        sigma = random.uniform(BLUR_SIGMA_MIN, BLUR_SIGMA_MAX)
        arr = cv2.GaussianBlur(arr, (0, 0), sigma)

    # 3 — Morfologia: erosão XOR dilatação, kernel e iterações variáveis
    if random.random() < MORFO_PROB:
        k_size = random.randint(MORFO_K_MIN, MORFO_K_MAX)
        iteracoes = random.randint(MORFO_ITER_MIN, MORFO_ITER_MAX)
        kernel = np.ones((k_size, k_size), np.uint8)
        if random.random() < 0.5:
            arr = cv2.erode(arr, kernel, iterations=iteracoes)
        else:
            arr = cv2.dilate(arr, kernel, iterations=iteracoes)

    # 4 — Ruído gaussiano: de quase imperceptível a muito ruidoso
    if random.random() < RUIDO_PROB:
        std = random.uniform(RUIDO_STD_MIN, RUIDO_STD_MAX)
        ruido = np.random.normal(0, std * 255, arr.shape).astype(np.float32)
        arr = np.clip(arr.astype(np.float32) + ruido, 0, 255).astype(np.uint8)

    # Converter de volta para imagem PIL
    nova_imagem = Image.fromarray(arr, mode="RGB")
    return nova_imagem, bboxes
