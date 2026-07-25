"""
inference.py
============
Pipeline completo: detector (YOLO, bbox de caractere) -> crop -> classificador
(ResNet-18, kanji N1) -> kanji previsto.

O classificador é treinado sobre as 1232 classes N1 (CLASSIFIER_KANJI_LEVEL="n1"
em config.py) mais uma classe extra CLF_OUTROS_LABEL ("OUTROS") — hiragana/
katakana, kanji fora do N1, letras latinas e
ruído/fundo vazio. `det.kanji` pode vir literalmente como "OUTROS" quando o
classificador rejeita a detecção por não parecer um kanji N1 de verdade — não
existe um filtro de codepoint em separado, a rejeição acontece no próprio
classificador.

Uso:
    from src.pipeline.inference import Pipeline

    pipeline = Pipeline()
    deteccoes = pipeline.predict(frame_bgr)  # frame_bgr: numpy array (H, W, 3) BGR
    for det in deteccoes:
        print(det.kanji, det.confianca_det, det.confianca_cls)
"""

import os
from dataclasses import dataclass

import torch
from PIL import Image

from config import (
    DETECTOR_WEIGHTS_PATH, CLF_WEIGHTS_PATH,
    PIPELINE_MIN_BBOX_HEIGHT, PIPELINE_BBOX_PADDING,
    PIPELINE_DET_CONF, PIPELINE_DET_IOU, PIPELINE_DET_MAX_DET,
    CLF_OUTROS_LABEL,
)
from src.classifier.model import build_model
from src.classifier.dataset import build_transform


@dataclass
class Deteccao:
    bbox: tuple           # (x1, y1, x2, y2) em pixels, no frame original (sem padding)
    kanji: str
    codepoint: str
    confianca_det: float  # confianca do detector (YOLO)
    confianca_cls: float  # confianca do classificador (softmax do topo)

def load_detector(weights_path: str = None):
    from ultralytics import YOLO
    weights_path = weights_path or DETECTOR_WEIGHTS_PATH
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Pesos do detector nao encontrados: {weights_path}\n"
            f"Treine o detector (notebooks/01_detector_train.ipynb) ou ajuste "
            f"KD_DETECTOR_WEIGHTS_PATH."
        )
    return YOLO(weights_path)


def load_classifier(weights_path: str = None, device: str = "cpu"):
    weights_path = weights_path or CLF_WEIGHTS_PATH
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Pesos do classificador nao encontrados: {weights_path}\n"
            f"Treine o classificador (notebooks/02_classifier_train.ipynb), baixe "
            f"o best.pt resultante e coloque nesse caminho (ou ajuste KD_CLF_WEIGHTS_PATH)."
        )
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    classes = checkpoint["classes"]
    # stem_leve muda stride/Identity (nao pesa no state_dict) -- se nao ler do
    # proprio checkpoint, a arquitetura reconstruida pode nao bater com a que
    # treinou os pesos (CLF_STEM_LEVE local pode divergir do valor usado no
    # treino, ex: setado so via env var na sessao do Kaggle). Checkpoints
    # salvos antes desse campo existir (default False) foram todos treinados
    # sem stem_leve, entao o fallback e' correto pra eles.
    stem_leve = checkpoint.get("stem_leve", False)
    model = build_model(num_classes=len(classes), stem_leve=stem_leve)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, classes


def codepoint_to_kanji(codepoint_str: str) -> str:
    if codepoint_str == CLF_OUTROS_LABEL:
        return codepoint_str
    return chr(int(codepoint_str.replace("U+", ""), 16))

def _expandir_bbox(x1, y1, x2, y2, margem, largura_max, altura_max):
    """Expande a bbox em `margem` proporcional (ex: 0.10 = +10%), clampeando aos limites do frame."""
    w, h = x2 - x1, y2 - y1
    pad_x, pad_y = w * margem, h * margem
    return (
        max(0, int(x1 - pad_x)),
        max(0, int(y1 - pad_y)),
        min(largura_max, int(x2 + pad_x)),
        min(altura_max, int(y2 + pad_y)),
    )

class Pipeline:
    """Agrupa detector + classificador já carregados, prontos pra rodar por frame."""

    def __init__(self, detector_weights: str = None, classifier_weights: str = None, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.detector = load_detector(detector_weights)
        self.classifier, self.classes = load_classifier(classifier_weights, self.device)
        self.transform = build_transform()

    def predict(self, frame_bgr,
                conf_det: float = PIPELINE_DET_CONF,
                iou_det: float = PIPELINE_DET_IOU) -> list:
        """
        Roda o pipeline completo num frame (numpy array BGR, ex: vindo do OpenCV/mss).
        Retorna uma lista de `Deteccao`, uma por bbox válida (>= PIPELINE_MIN_BBOX_HEIGHT px).
        """
        h_frame, w_frame = frame_bgr.shape[:2]

        results = self.detector(
            frame_bgr, conf=conf_det, iou=iou_det,
            agnostic_nms=True, max_det=PIPELINE_DET_MAX_DET, verbose=False,
        )[0]

        frame_rgb = Image.fromarray(frame_bgr[:, :, ::-1])  # BGR -> RGB pra PIL

        deteccoes = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            det_conf = float(box.conf[0])

            if (y2 - y1) < PIPELINE_MIN_BBOX_HEIGHT:
                continue

            xe1, ye1, xe2, ye2 = _expandir_bbox(
                x1, y1, x2, y2, PIPELINE_BBOX_PADDING, w_frame, h_frame
            )
            crop = frame_rgb.crop((xe1, ye1, xe2, ye2)).convert("L")

            x = self.transform(crop).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.classifier(x)
                probs = torch.softmax(logits, dim=1)
                cls_conf, pred_idx = probs.max(dim=1)

            codepoint = self.classes[pred_idx.item()]

            deteccoes.append(Deteccao(
                bbox=(int(x1), int(y1), int(x2), int(y2)),
                kanji=codepoint_to_kanji(codepoint),
                codepoint=codepoint,
                confianca_det=det_conf,
                confianca_cls=float(cls_conf.item()),
            ))

        return deteccoes


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Uso: python -m src.pipeline.inference <caminho_da_imagem>")
        sys.exit(1)

    import cv2

    img_path = sys.argv[1]
    frame = cv2.imread(img_path)
    if frame is None:
        raise FileNotFoundError(f"Nao foi possivel ler a imagem: {img_path}")

    print("[INFO] Carregando pipeline...")
    pipeline = Pipeline()
    print(f"[INFO] Rodando inferencia em {img_path}...")
    deteccoes = pipeline.predict(frame)

    print(f"\n[INFO] {len(deteccoes)} deteccoes:")
    for det in deteccoes:
        print(f"  {det.kanji} ({det.codepoint}) | det={det.confianca_det:.2f} cls={det.confianca_cls:.2f} | bbox={det.bbox}")
