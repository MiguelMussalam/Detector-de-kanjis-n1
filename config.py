import os

# ---------------------------------------------------------------------------
# Helper: lê variável de ambiente com fallback para o valor padrão.
# Faz cast automático para o mesmo tipo do default.
# ---------------------------------------------------------------------------

def _env(name: str, default):
    """
    Retorna o valor de uma variável de ambiente, convertendo para o tipo de
    `default`. Se a variável não existir, retorna `default` sem alteração.
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    t = type(default)
    if t is bool:
        return raw.lower() in ("1", "true", "yes")
    return t(raw)


# ---------------------------------------------------------------------------
# Caminhos base (não tunáveis — não fazem sentido como env vars)
# ---------------------------------------------------------------------------

ROOT_DIR        = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR      = os.path.join(ROOT_DIR, "assets")
DATA_DIR        = os.path.join(ROOT_DIR, "data")

# Fontes
FONTS_DIR       = os.path.join(ASSETS_DIR, "fonts")

# Manga109
MANGA109_DIR    = os.path.join(DATA_DIR, "raw", "Manga109")

def _buscar_manga109_images():
    # Caminho padrão local
    caminho_local = os.path.join(MANGA109_DIR, "images")
    if os.path.exists(caminho_local):
        return caminho_local

    # Busca dinâmica no ambiente Kaggle
    kaggle_input = "/kaggle/input"
    if os.path.exists(kaggle_input):
        for root, dirs, _ in os.walk(kaggle_input):
            if "images" in dirs and "manga109" in root.lower():
                found_path = os.path.join(root, "images")
                print(f"[INFO] Manga109 images encontradas no Kaggle: {found_path}")
                return found_path
    return caminho_local

MANGA109_IMAGES = _buscar_manga109_images()

# Dataset sintético
SYNTHETIC_DIR   = os.path.join(DATA_DIR, "synthetic")
PAGES_DIR       = os.path.join(SYNTHETIC_DIR, "pages")

# Dataset YOLO (estrutura images/labels train/val)
DATASET_DIR     = os.path.join(DATA_DIR, "dataset")
TRAIN_IMG_DIR   = os.path.join(DATASET_DIR, "images", "train")
VAL_IMG_DIR     = os.path.join(DATASET_DIR, "images", "val")
TRAIN_LBL_DIR   = os.path.join(DATASET_DIR, "labels", "train")
VAL_LBL_DIR     = os.path.join(DATASET_DIR, "labels", "val")

# URLs externas (não tunáveis)
KANJI_DATA_URL  = "https://raw.githubusercontent.com/davidluzgouveia/kanji-data/master/kanji.json"
KANJI_DATA_CACHE = os.path.join(ASSETS_DIR, "kanji.json")
FONTES_URL = {
    "Shippori Antique":         "https://raw.githubusercontent.com/fontdasu/ShipporiAntique/master/fonts/ttf/ShipporiAntique-Regular.ttf",
    "BIZ-UDPGothic-Regular":    "https://raw.githubusercontent.com/google/fonts/main/ofl/bizudpgothic/BIZUDPGothic-Regular.ttf",
    "BIZ-UDPMincho-Regular":    "https://raw.githubusercontent.com/google/fonts/main/ofl/bizudpmincho/BIZUDPMincho-Regular.ttf",
    "Klee-One-Regular":         "https://raw.githubusercontent.com/google/fonts/main/ofl/kleeone/KleeOne-Regular.ttf",
    "Hina-Mincho-Regular":      "https://raw.githubusercontent.com/google/fonts/main/ofl/hinamincho/HinaMincho-Regular.ttf",
    "Yusei-Magic-Regular":      "https://raw.githubusercontent.com/google/fonts/main/ofl/yuseimagic/YuseiMagic-Regular.ttf",
    "Dela-Gothic-One":          "https://raw.githubusercontent.com/google/fonts/main/ofl/delagothicone/DelaGothicOne-Regular.ttf",
    "Reggae-One":               "https://raw.githubusercontent.com/google/fonts/main/ofl/reggaeone/ReggaeOne-Regular.ttf",
}

# ---------------------------------------------------------------------------
# Parâmetros tunáveis — lidos de variáveis de ambiente (com fallback padrão)
# ---------------------------------------------------------------------------

# Dataset
VAL_RATIO       = _env("KD_VAL_RATIO",    0.15)  # fração para validação
CROP_SIZE       = _env("KD_CROP_SIZE",    640)   # tamanho do crop em pixels
PAGES_AMOUNT    = _env("KD_PAGES_AMOUNT", 20) # total de páginas sintéticas (aumentado de 12000)

# Posicionamento de texto
GAP_CHAR        = _env("KD_GAP_CHAR",    4)
GAP_COL         = _env("KD_GAP_COL",     8)
LIMITE_DESVIO_REGIAO   = _env("KD_LIMITE_DESVIO",    25)
MAX_TENTATIVAS_POSICAO = _env("KD_MAX_TENTATIVAS",    10)
BBOX_MARGEM            = _env("KD_BBOX_MARGEM",      0.10)  # expansao proporcional das bboxes (ex: 0.10 = +10%)

# Treino YOLO
YOLO_MODEL      = _env("KD_YOLO_MODEL",  "yolo26n.pt")
EPOCHS          = _env("KD_EPOCHS",      50)
IMGSZ           = _env("KD_IMGSZ",       640)
BATCH           = _env("KD_BATCH",       16)
KAGGLE_WORKERS  = _env("KD_KAGGLE_WORKERS", 2)   # T4/P100 têm 2 vCPUs
LOCAL_WORKERS   = _env("KD_LOCAL_WORKERS",  4)
PROJECT_NAME    = _env("KD_PROJECT_NAME", "kanji_detector")
KAGGLE_DATASET  = _env("KD_KAGGLE_DATASET", "kanji-detector-dataset")

# Imagens negativas
PROB_NEGATIVA   = _env("KD_PROB_NEGATIVA", 0.20)  # 20% das imagens sem nenhum caractere

# Quantidade de blocos por imagem
N_BLOCOS_MIN    = _env("KD_N_BLOCOS_MIN",  3)
N_BLOCOS_MAX    = _env("KD_N_BLOCOS_MAX",  8)

# Tamanho de cada bloco
N_COLUNAS_MIN        = _env("KD_N_COLUNAS_MIN",        2)
N_COLUNAS_MAX        = _env("KD_N_COLUNAS_MAX",        6)
CHARS_POR_COLUNA_MIN = _env("KD_CHARS_POR_COLUNA_MIN", 6)
CHARS_POR_COLUNA_MAX = _env("KD_CHARS_POR_COLUNA_MAX", 16)

# Blur
BLUR_PROB       = _env("KD_BLUR_PROB",       0.50)
BLUR_SIGMA_MIN  = _env("KD_BLUR_SIGMA_MIN",  0.1)
BLUR_SIGMA_MAX  = _env("KD_BLUR_SIGMA_MAX",  2.5)

# Morfologia
MORFO_PROB      = _env("KD_MORFO_PROB",      0.40)
MORFO_K_MIN     = _env("KD_MORFO_K_MIN",     2)
MORFO_K_MAX     = _env("KD_MORFO_K_MAX",     4)
MORFO_ITER_MIN  = _env("KD_MORFO_ITER_MIN",  1)
MORFO_ITER_MAX  = _env("KD_MORFO_ITER_MAX",  3)

# Ruído
RUIDO_PROB      = _env("KD_RUIDO_PROB",      0.60)
RUIDO_STD_MIN   = _env("KD_RUIDO_STD_MIN",   0.01)
RUIDO_STD_MAX   = _env("KD_RUIDO_STD_MAX",   0.25)

# Rotação
ROTACAO_PROB    = _env("KD_ROTACAO_PROB",    0.40)
ROTACAO_MAX     = _env("KD_ROTACAO_MAX",     3.0)  # graus (±)
