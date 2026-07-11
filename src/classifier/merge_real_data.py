"""
merge_real_data.py
===================
Mescla os crops REAIS de OUTROS gerados por manga109_align.py (em
data/classifier_real/) com o dataset sintético (em data/classifier/), copiando
os arquivos para a mesma pasta OUTROS. O ImageFolder do dataset.py passa a
enxergar os dois juntos automaticamente, sem nenhuma mudança de código.

Só OUTROS é mesclado -- N1 real foi descartado de propósito:
  1. Cobertura real por classe é baixa demais pra servir de treino (a maioria
     das classes que aparecem tem só 1-3 exemplos, kanji raro é raro na
     língua, não só no corpus).
  2. Pior: erro residual de alinhamento pode colocar crop de UM kanji na
     pasta de OUTRO -- numa classe com só 1-3 exemplos reais, isso não é
     ruído irrelevante, pode ser o único exemplo real ensinando a associação
     errada. OUTROS não tem esse risco na mesma escala (é um "guarda-chuva"
     com milhares de caracteres diferentes; um crop mal alinhado ali é só
     mais um exemplo entre milhares, não domina a classe).
OUTROS real e subamostrado (era muito maior que o teto desejado) para nao
desbalancear o treino: OUTROS_TRAIN_MAX / OUTROS_VAL_MAX abaixo.

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
    if not os.path.isdir(MANGA109_ALIGN_TRAIN_DIR):
        print(f"[AVISO] {MANGA109_ALIGN_TRAIN_DIR} nao existe -- alinhamento nao rodou "
              f"(ou falhou) antes deste merge. Seguindo so com o dataset sintetico.")
        return

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

    print(f"[INFO] OUTROS real mesclado: {total_outros_train} treino / {total_outros_val} val "
          f"(subamostrado de ate {OUTROS_TRAIN_MAX}/{OUTROS_VAL_MAX})")
    print("[INFO] Merge completo (N1 real nao e usado -- ver docstring).")


if __name__ == "__main__":
    main()
