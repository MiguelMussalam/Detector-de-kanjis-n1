"""
merge_real_data.py
===================
Mescla os crops reais gerados por manga109_align.py (em data/classifier_real/)
com o dataset sintético (em data/classifier/), copiando os arquivos para as
mesmas pastas por classe. O ImageFolder do dataset.py passa a enxergar os dois
juntos automaticamente, sem nenhuma mudança de código.

- Todos os crops N1 reais são copiados (nao ha excesso, ao contrario do OUTROS).
- OUTROS real e subamostrado (era 40x maior que o N1 real) para nao desbalancear
  o treino: MANGA109_ALIGN_OUTROS_TRAIN_MAX / _VAL_MAX em config.py.

Uso:
    python -m src.classifier.merge_real_data
"""

import os
import random
import shutil

from config import (
    CLASSIFIER_TRAIN_DIR, CLASSIFIER_VAL_DIR,
    MANGA109_ALIGN_TRAIN_DIR, MANGA109_ALIGN_VAL_DIR,
    CLF_OUTROS_LABEL,
)

OUTROS_TRAIN_MAX = 20000
OUTROS_VAL_MAX = 5000
SEED = 2024


def copiar_classe(src_dir: str, dst_dir: str, limite: int = None, seed: int = SEED) -> int:
    """Copia os arquivos de src_dir pra dst_dir, com prefixo 'real_' para nao
    colidir com nomes sinteticos. Se limite for dado, subamostra aleatoriamente."""
    if not os.path.isdir(src_dir):
        return 0
    arquivos = os.listdir(src_dir)
    if limite is not None and len(arquivos) > limite:
        rng = random.Random(seed)
        arquivos = rng.sample(arquivos, limite)

    os.makedirs(dst_dir, exist_ok=True)
    for fname in arquivos:
        shutil.copy(
            os.path.join(src_dir, fname),
            os.path.join(dst_dir, f"real_{fname}"),
        )
    return len(arquivos)


def main():
    # --- N1: copia tudo, sem subamostrar ---
    # Uniao das classes com dado real no treino OU no val (um kanji raro pode
    # so ter aparecido nos 10 volumes de val e nenhum dos 99 de treino).
    classes_train = {d for d in os.listdir(MANGA109_ALIGN_TRAIN_DIR) if d.startswith("U+")}
    classes_val = {d for d in os.listdir(MANGA109_ALIGN_VAL_DIR) if d.startswith("U+")}
    classes_n1 = sorted(classes_train | classes_val)
    print(f"[INFO] {len(classes_n1)} classes N1 com dado real "
          f"({len(classes_train)} no treino, {len(classes_val)} no val)")

    total_n1_train = 0
    total_n1_val = 0
    for classe in classes_n1:
        total_n1_train += copiar_classe(
            os.path.join(MANGA109_ALIGN_TRAIN_DIR, classe),
            os.path.join(CLASSIFIER_TRAIN_DIR, classe),
        )
        total_n1_val += copiar_classe(
            os.path.join(MANGA109_ALIGN_VAL_DIR, classe),
            os.path.join(CLASSIFIER_VAL_DIR, classe),
        )

    # --- OUTROS: subamostra ---
    total_outros_train = copiar_classe(
        os.path.join(MANGA109_ALIGN_TRAIN_DIR, CLF_OUTROS_LABEL),
        os.path.join(CLASSIFIER_TRAIN_DIR, CLF_OUTROS_LABEL),
        limite=OUTROS_TRAIN_MAX,
    )
    total_outros_val = copiar_classe(
        os.path.join(MANGA109_ALIGN_VAL_DIR, CLF_OUTROS_LABEL),
        os.path.join(CLASSIFIER_VAL_DIR, CLF_OUTROS_LABEL),
        limite=OUTROS_VAL_MAX,
    )

    print(f"[INFO] N1 real mesclado:     {total_n1_train} treino / {total_n1_val} val")
    print(f"[INFO] OUTROS real mesclado: {total_outros_train} treino / {total_outros_val} val "
          f"(subamostrado de ate {OUTROS_TRAIN_MAX}/{OUTROS_VAL_MAX})")
    print("[INFO] Merge completo.")


if __name__ == "__main__":
    main()
