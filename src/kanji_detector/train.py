"""
train.py
========
Treina o detector de kanjis com YOLOv26n.

Roda localmente e no Kaggle automaticamente:
  - Detecta ambiente Kaggle via variavel de ambiente KAGGLE_DATA_PROXY_TOKEN
    ou pela existencia de /kaggle/input/
  - Ajusta paths do dataset.yaml e numero de workers

Uso (local):
    python -m src.kanji_detector.train

Uso (Kaggle notebook):
    !python /kaggle/working/src/kanji_detector/train.py
"""

import os
import sys
import yaml

# ---------------------------------------------------------------------------
# Deteccao de ambiente
# ---------------------------------------------------------------------------

IS_KAGGLE = (
    os.path.exists("/kaggle/input") or
    "KAGGLE_DATA_PROXY_TOKEN" in os.environ or
    "KAGGLE_KERNEL_RUN_TYPE" in os.environ
)

if IS_KAGGLE:
    print("[INFO] Ambiente Kaggle detectado.")
    # Adiciona o diretorio de trabalho ao path para imports funcionarem
    sys.path.insert(0, "/kaggle/working")
else:
    print("[INFO] Ambiente local detectado.")

from config import (
    DATASET_DIR, EPOCHS, IMGSZ, BATCH,
    KAGGLE_WORKERS, LOCAL_WORKERS, PROJECT_NAME,
    KAGGLE_DATASET, YOLO_MODEL, ROOT_DIR,
)


# ---------------------------------------------------------------------------
# Resolucao de paths
# ---------------------------------------------------------------------------

def resolver_paths_kaggle(kaggle_dataset_name: str) -> str:
    """
    No Kaggle, o dataset fica em /kaggle/input/<dataset-name>/.
    Esta funcao gera um dataset.yaml temporario com os paths corretos
    e retorna o caminho para ele.
    """
    kaggle_dataset_dir = f"/kaggle/input/{kaggle_dataset_name}"

    if not os.path.exists(kaggle_dataset_dir):
        raise FileNotFoundError(
            f"Dataset nao encontrado em: {kaggle_dataset_dir}\n"
            f"Certifique-se de ter adicionado o dataset '{kaggle_dataset_name}' "
            f"ao notebook Kaggle."
        )

    yaml_path = "/kaggle/working/dataset.yaml"
    config = {
        "path": kaggle_dataset_dir,
        "train": os.path.join("images", "train"),
        "val":   os.path.join("images", "val"),
        "nc":    1,
        "names": {0: "kanji"},
    }
    with open(yaml_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"[INFO] dataset.yaml Kaggle gerado em: {yaml_path}")
    print(f"[INFO] Dataset path: {kaggle_dataset_dir}")
    return yaml_path


def resolver_paths_local() -> str:
    """Retorna o caminho do dataset.yaml gerado pelo generate_pages.py."""
    yaml_path = os.path.join(DATASET_DIR, "dataset.yaml")
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(
            f"dataset.yaml nao encontrado em: {yaml_path}\n"
            "Execute primeiro: python -m src.kanji_detector.generate_pages"
        )
    return yaml_path


# ---------------------------------------------------------------------------
# Treino
# ---------------------------------------------------------------------------

def treinar():
    from ultralytics import YOLO

    # Resolve paths e workers conforme ambiente
    if IS_KAGGLE:
        workers = KAGGLE_WORKERS
        # Modelo: primeiro tenta o arquivo local, depois deixa a ultralytics baixar
        model_path = os.path.join("/kaggle/working", YOLO_MODEL)
        if not os.path.exists(model_path):
            model_path = YOLO_MODEL  # ultralytics faz o download automatico

        # Se o dataset foi gerado localmente na sessao do Kaggle (Opcao 1)
        dataset_local_yaml = os.path.join(DATASET_DIR, "dataset.yaml")
        # Se o dataset foi adicionado como Input pronto no Kaggle (Opcao 2)
        kaggle_input_dir = f"/kaggle/input/{KAGGLE_DATASET}"

        if os.path.exists(kaggle_input_dir):
            print(f"[INFO] Utilizando dataset de entrada do Kaggle: {kaggle_input_dir}")
            data_yaml = resolver_paths_kaggle(KAGGLE_DATASET)
        elif os.path.exists(dataset_local_yaml):
            print(f"[INFO] Utilizando dataset gerado na sessao (/kaggle/working): {dataset_local_yaml}")
            data_yaml = dataset_local_yaml
        else:
            raise FileNotFoundError(
                f"Nao foi encontrado nenhum dataset de treino no Kaggle!\n"
                f"1. Se gerou no Kaggle, certifique-se de ter rodado o generate_pages.py primeiro.\n"
                f"2. Se enviou o dataset pronto, verifique se o adicionou como Input com o nome '{KAGGLE_DATASET}'."
            )
    else:
        data_yaml = resolver_paths_local()
        workers   = LOCAL_WORKERS
        model_path = os.path.join(ROOT_DIR, YOLO_MODEL)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modelo nao encontrado: {model_path}")

    print(f"[INFO] Modelo:   {model_path}")
    print(f"[INFO] Data:     {data_yaml}")
    print(f"[INFO] Epochs:   {EPOCHS}")
    print(f"[INFO] Batch:    {BATCH}")
    print(f"[INFO] Workers:  {workers}")
    print(f"[INFO] Img size: {IMGSZ}")

    model = YOLO(model_path)

    results = model.train(
        data     = data_yaml,
        epochs   = EPOCHS,
        imgsz    = IMGSZ,
        batch    = BATCH,
        workers  = workers,
        project  = PROJECT_NAME,
        name     = "run1",
        exist_ok = True,   # permite retomar runs com mesmo nome
        # Kaggle: salva checkpoints no working dir
        save     = True,
    )

    print("[INFO] Treino concluido!")
    print(f"[INFO] Resultados em: {results.save_dir}")
    return results


if __name__ == "__main__":
    treinar()
