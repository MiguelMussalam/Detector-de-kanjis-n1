"""
train.py
========
Treina o detector de caracteres com YOLO.

Assume que o dataset (data.yaml + imagens/labels no formato Ultralytics YOLO)
já foi preparado — essa responsabilidade é do notebook de treino.

Roda localmente e no Kaggle automaticamente:
  - Detecta ambiente Kaggle via variavel de ambiente KAGGLE_DATA_PROXY_TOKEN
    ou pela existencia de /kaggle/input/
  - Ajusta numero de workers conforme o ambiente

Uso (local):
    python -m src.detector.train

Uso (Kaggle notebook):
    !python -m src.detector.train
"""

import os
import sys

IS_KAGGLE = (
    os.path.exists("/kaggle/input") or
    "KAGGLE_DATA_PROXY_TOKEN" in os.environ or
    "KAGGLE_KERNEL_RUN_TYPE" in os.environ
)

if IS_KAGGLE:
    print("[INFO] Ambiente Kaggle detectado.")
    sys.path.insert(0, "/kaggle/working")
else:
    print("[INFO] Ambiente local detectado.")

from config import (
    DATA_YAML, EPOCHS, IMGSZ, BATCH,
    KAGGLE_WORKERS, LOCAL_WORKERS, PROJECT_NAME,
    YOLO_MODEL, ROOT_DIR,
)

def main():
    from ultralytics import YOLO

    if not os.path.exists(DATA_YAML):
        raise FileNotFoundError(
            f"data.yaml nao encontrado em: {DATA_YAML}\n"
            f"Prepare o dataset (celulas do notebook) antes de treinar, ou "
            f"ajuste o caminho via variavel de ambiente KD_DATA_YAML."
        )

    if IS_KAGGLE:
        workers = KAGGLE_WORKERS
        model_path = os.path.join("/kaggle/working", YOLO_MODEL)
        if not os.path.exists(model_path):
            model_path = YOLO_MODEL  # ultralytics faz o download automatico
    else:
        workers = LOCAL_WORKERS
        model_path = os.path.join(ROOT_DIR, YOLO_MODEL)
        if not os.path.exists(model_path):
            model_path = YOLO_MODEL  # ultralytics faz o download automatico

    print(f"[INFO] Modelo:   {model_path}")
    print(f"[INFO] Data:     {DATA_YAML}")
    print(f"[INFO] Epochs:   {EPOCHS}")
    print(f"[INFO] Batch:    {BATCH}")
    print(f"[INFO] Workers:  {workers}")
    print(f"[INFO] Img size: {IMGSZ}")

    model = YOLO(model_path)

    # Caminho absoluto para o project: versoes recentes do Ultralytics podem
    # prefixar um "runs/<task>/" na frente de um project relativo, o que
    # quebra qualquer calculo de caminho feito por fora (ex: no notebook).
    # Absoluto remove essa ambiguidade.
    project_dir = os.path.join(ROOT_DIR, PROJECT_NAME)

    results = model.train(
        data     = DATA_YAML,
        epochs   = EPOCHS,
        imgsz    = IMGSZ,
        batch    = BATCH,
        workers  = workers,
        project  = project_dir,
        name     = "run",
        exist_ok = True,
        save     = True,
    )

    print("[INFO] Treino concluido!")
    print(f"[INFO] Resultados em: {results.save_dir}")

    # Grava o caminho real de saida num arquivo marcador, para o notebook
    # (ou qualquer outro consumidor) ler em vez de tentar reconstruir/adivinhar
    # o caminho de fora — evita qualquer divergencia de versao do Ultralytics.
    marcador_path = os.path.join(ROOT_DIR, "ultimo_run_detector.txt")
    with open(marcador_path, "w", encoding="utf-8") as f:
        f.write(str(results.save_dir))
    print(f"[INFO] Caminho gravado em: {marcador_path}")

    return results


if __name__ == "__main__":
    main()
