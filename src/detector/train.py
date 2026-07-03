"""
train.py
========
Treina o detector de kanjis com YOLOv26n.

Roda localmente e no Kaggle automaticamente:
  - Detecta ambiente Kaggle via variavel de ambiente KAGGLE_DATA_PROXY_TOKEN
    ou pela existencia de /kaggle/input/
  - Ajusta paths do dataset.yaml e numero de workers

Uso (local):
    python -m src.detector.train

Uso (Kaggle notebook):
    !python /kaggle/working/src/detector/train.py
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
    MODO_TREINO, MANGA109_YOLO_DIR, SYNTHETIC_YOLO_DIR, DATA_DIR,
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
        "train": "images/train",
        "val":   "images/val",
        "nc":    1,
        "names": ["text"],
    }
    with open(yaml_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"[INFO] dataset.yaml Kaggle gerado em: {yaml_path}")
    print(f"[INFO] Dataset path: {kaggle_dataset_dir}")
    return yaml_path


def resolver_dataset(modo: str) -> str:
    if modo == "manga109":
        yaml_path = os.path.join(MANGA109_YOLO_DIR, "data.yaml")
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(
                "Dataset Manga109 não encontrado. "
                "Execute python -m src.detector.prepare_manga109 primeiro."
            )
        return yaml_path

    elif modo == "dado_sintetico":
        yaml_path = os.path.join(SYNTHETIC_YOLO_DIR, "data.yaml")
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(
                "Dataset sintético não encontrado. "
                "Execute python -m src.detector.generate_pages primeiro."
            )
        return yaml_path

    elif modo == "misto":
        yaml_path = os.path.join(DATA_DIR, "misto_yolo", "data.yaml")
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(
                "Dataset misto não encontrado. "
                "Execute python -m src.detector.prepare_misto primeiro."
            )
        return yaml_path

    else:
        raise ValueError(f"MODO_TREINO inválido: '{modo}'. "
                         "Use 'manga109', 'dado_sintetico' ou 'misto'.")


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

        # Tenta resolver o dataset com base no MODO_TREINO local no Kaggle
        try:
            data_yaml = resolver_dataset(MODO_TREINO)
            print(f"[INFO] Utilizando dataset gerado na sessao para '{MODO_TREINO}': {data_yaml}")
        except FileNotFoundError:
            # Se não encontrar local, tenta usar o dataset do input do Kaggle
            kaggle_input_dir = f"/kaggle/input/{KAGGLE_DATASET}"
            if os.path.exists(kaggle_input_dir):
                print(f"[INFO] Utilizando dataset de entrada do Kaggle: {kaggle_input_dir}")
                data_yaml = resolver_paths_kaggle(KAGGLE_DATASET)
            else:
                raise FileNotFoundError(
                    f"Nao foi encontrado nenhum dataset de treino para o modo '{MODO_TREINO}' no Kaggle!\n"
                    f"1. Se gerou no Kaggle, certifique-se de ter rodado o script de geracao correspondente primeiro.\n"
                    f"2. Se enviou o dataset pronto, verifique se o adicionou como Input com o nome '{KAGGLE_DATASET}'."
                )
    else:
        data_yaml = resolver_dataset(MODO_TREINO)
        workers   = LOCAL_WORKERS
        model_path = os.path.join(ROOT_DIR, YOLO_MODEL)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modelo nao encontrado: {model_path}")

    print(f"[INFO] Modo Treino: {MODO_TREINO}")
    print(f"[INFO] Modelo:      {model_path}")
    print(f"[INFO] Data:        {data_yaml}")
    print(f"[INFO] Epochs:      {EPOCHS}")
    print(f"[INFO] Batch:       {BATCH}")
    print(f"[INFO] Workers:     {workers}")
    print(f"[INFO] Img size:    {IMGSZ}")

    model = YOLO(model_path)

    results = model.train(
        data     = data_yaml,
        epochs   = EPOCHS,
        imgsz    = IMGSZ,
        batch    = BATCH,
        workers  = workers,
        project  = PROJECT_NAME,
        name     = f"run_{MODO_TREINO}",
        exist_ok = True,   # permite retomar runs com mesmo nome
        # Kaggle: salva checkpoints no working dir
        save     = True,
    )

    print("[INFO] Treino concluido!")
    print(f"[INFO] Resultados em: {results.save_dir}")
    return results


if __name__ == "__main__":
    treinar()
