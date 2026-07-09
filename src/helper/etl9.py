"""
etl9.py
=======
Wrapper de leitura do ETL9 (ETL9B ou ETL9G), com filtragem para as classes do
classificador.

Usado no diagnóstico do classificador como proxy de generalização fora de domínio
(ETL9 é manuscrito, classificador foi treinado em impresso sintético — se a acurácia
for razoável, o modelo aprendeu features generalizáveis de kanji, não só de fonte).

Estrutura de pasta esperada (a biblioteca `etl_data_reader` é exigente com isso —
não é só jogar os arquivos soltos em ETL9_DIR). O nome da subpasta e dos arquivos
segue a numeração INTERNA da lib, que não bate com o nome oficial do AIST:

    ETL9_VERSION="ETL9B" (binário, 5 arquivos):
        data/etl9/euc_co59.dat
        data/etl9/ETL10/ETL10_1 .. ETL10_5

    ETL9_VERSION="ETL9G" (grayscale, 50 arquivos):
        data/etl9/euc_co59.dat
        data/etl9/ETL11/ETL11_01 .. ETL11_50

Se você baixou o zip oficial do AIST (que vem com pasta "ETL9G/" e arquivos
"ETL9G_01".."ETL9G_50"), é preciso renomear a pasta pra "ETL11" e cada arquivo
de "ETL9G_NN" para "ETL11_NN" antes de usar — a lib não reconhece o nome original.

Onde conseguir:
    http://etlcdb.db.aist.go.jp/ (registro gratuito, aprovação em 1-2 dias).
    O arquivo de código de caracteres (euc_co59.dat) não vem dentro do zip do
    ETL9 — é baixado separado, na página de downloads do ETL8/9 do AIST.
"""

import os
from typing import Set

import numpy as np
from PIL import Image as _PILImage

# etl_data_reader usa Image.ANTIALIAS, removido no Pillow >= 10 (era alias de LANCZOS).
# Monkey-patch aqui em vez de fixar uma versão antiga de Pillow no requirements.txt.
if not hasattr(_PILImage, "ANTIALIAS"):
    _PILImage.ANTIALIAS = _PILImage.LANCZOS

from config import ETL9_DIR, ETL9_VERSION


def _para_uint8(img: np.ndarray) -> np.ndarray:
    """
    O etl_data_reader retorna imagens float16 com um canal extra (H, W, 1),
    já normalizadas para [0, 1] quando o formato tem mais de 1 bit de profundidade
    (não é o caso do ETL9B, que é binário — mas tratamos os dois casos por segurança).
    Converte para uint8 (H, W) em [0, 255], formato que o resto do pipeline espera.

    Inverte a polaridade: o ETL9 vem com o traço do caractere claro sobre fundo
    escuro (verificado visualmente — média de pixel ~6/255, quase todo preto).
    O classificador só viu o oposto no treino sintético (traço escuro sobre papel
    claro, ver CLF_BG_VALUE_MIN/MAX em config.py). Sem essa inversão, a imagem
    chega com a polaridade errada e o modelo erra quase tudo.
    """
    img = np.asarray(img)
    if img.ndim == 3 and img.shape[-1] == 1:
        img = img.squeeze(-1)
    if img.dtype != np.uint8:
        maximo = float(img.max()) if img.size else 1.0
        if maximo <= 1.0:
            img = img * 255.0
        img = np.clip(img, 0, 255).astype(np.uint8)
    img = 255 - img
    return img


def load_etl9(target_kanjis: Set[str] = None, resize: tuple = (64, 64)):
    """
    Carrega o ETL9B e retorna arrays prontos para avaliação.

    Args:
        target_kanjis: Conjunto de caracteres a incluir. Se None, carrega tudo
                       (só kanji — hiragana/katakana já são excluídos pela lib).
                       Passar aqui as classes do classificador filtra automaticamente
                       para o que ele conhece.
        resize:        Tamanho de saída (H, W). Padrão bate com input do classificador.

    Returns:
        images:  np.ndarray (N, H, W) uint8, grayscale
        labels:  list[str] de tamanho N, cada elemento é o caractere kanji
        stats:   dict com {"n_samples", "n_classes_encontradas", "n_classes_faltando", ...}
    """
    from etldr.etl_data_reader import ETLDataReader
    from etldr.etl_data_names import ETLDataNames
    from etldr.etl_character_groups import ETLCharacterGroups

    if not os.path.isdir(ETL9_DIR):
        raise FileNotFoundError(
            f"Diretório ETL9 não encontrado: {ETL9_DIR}\n"
            f"Veja o docstring deste módulo para a estrutura de pastas exata "
            f"esperada (inclui euc_co59.dat e a subpasta ETL10/)."
        )
    if not os.path.isfile(os.path.join(ETL9_DIR, "euc_co59.dat")):
        raise FileNotFoundError(
            f"Arquivo 'euc_co59.dat' não encontrado em {ETL9_DIR}.\n"
            f"Esse arquivo é obrigatório (tabela de códigos) e é baixado separado "
            f"dos arquivos de dados no site do AIST."
        )

    reader = ETLDataReader(ETL9_DIR)
    dataset_enum = getattr(ETLDataNames, ETL9_VERSION)

    print(f"[INFO] Carregando {ETL9_VERSION} de {ETL9_DIR}...")
    # read_dataset_part (não read_dataset_whole!) le so a parte pedida — read_dataset_whole
    # iteraria por TODOS os 11 datasets ETL1..ETL9G e quebraria pelos que nao existem aqui.
    images_raw, labels_raw = reader.read_dataset_part(
        dataset_enum,
        [ETLCharacterGroups.kanji],  # só kanji, exclui hiragana/katakana
        resize=resize,
    )
    labels = list(labels_raw)
    images = np.stack([_para_uint8(img) for img in images_raw]) if len(images_raw) else np.empty((0,) + resize, dtype=np.uint8)

    print(f"[INFO] Total carregado: {len(images)} amostras, "
          f"{len(set(labels))} classes únicas")

    if target_kanjis is not None:
        target_set = set(target_kanjis)
        mask = np.array([lbl in target_set for lbl in labels])
        images = images[mask]
        labels = [lbl for lbl in labels if lbl in target_set]

        n_encontradas = len(set(labels))
        n_faltando = len(target_set) - n_encontradas
        stats = {
            "n_samples": len(images),
            "n_classes_encontradas": n_encontradas,
            "n_classes_faltando": n_faltando,
            "n_classes_target": len(target_set),
        }
        print(f"[INFO] Após filtro: {len(images)} amostras, "
              f"{n_encontradas}/{len(target_set)} classes cobertas "
              f"({n_faltando} classes N1 não existem no ETL9)")
    else:
        stats = {
            "n_samples": len(images),
            "n_classes_encontradas": len(set(labels)),
        }

    return images, labels, stats


if __name__ == "__main__":
    from src.helper.kanjis import get_kanjis

    n1 = get_kanjis("n1")
    print(f"Lista N1 tem {len(n1)} kanji")

    images, labels, stats = load_etl9(target_kanjis=set(n1))

    print("\n=== Cobertura do ETL9 para N1 ===")
    print(f"Total N1:                    {stats['n_classes_target']}")
    print(f"N1 presentes no ETL9:        {stats['n_classes_encontradas']}")
    print(f"N1 faltando no ETL9:         {stats['n_classes_faltando']}")
    print(f"Amostras totais utilizáveis: {stats['n_samples']}")
    print(f"Média por classe:            {stats['n_samples']/max(1,stats['n_classes_encontradas']):.1f}")
