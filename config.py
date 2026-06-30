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
VAL_SPLIT       = _env("KD_VAL_SPLIT",    0.2)   # fração para validação
CROP_SIZE       = _env("KD_CROP_SIZE",    640)   # tamanho do crop em pixels
PAGES_AMOUNT    = _env("KD_PAGES_AMOUNT", 12000) # total de páginas sintéticas

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

# Degradações 2D — ordem de aplicação: geometria → tinta → scanner

# Morfologia (simulação de tinta)
EROSAO_PROB         = _env("KD_EROSAO_PROB",      0.3)
DILATACAO_PROB      = _env("KD_DILATACAO_PROB",   0.3)
MORFO_KERNEL        = _env("KD_MORFO_KERNEL",     2)
MORFO_ITERACOES     = _env("KD_MORFO_ITERACOES",  1)

# Ruído de scanner
GAUSS_NOISE_PROB    = _env("KD_GAUSS_NOISE_PROB",    0.5)
GAUSS_NOISE_STD_MIN = _env("KD_GAUSS_NOISE_STD_MIN", 0.05)
GAUSS_NOISE_STD_MAX = _env("KD_GAUSS_NOISE_STD_MAX", 0.15)
SALT_PEPPER_PROB    = _env("KD_SALT_PEPPER_PROB",    0.4)
BLUR_PROB           = _env("KD_BLUR_PROB",           0.35)
BLUR_KERNEL         = _env("KD_BLUR_KERNEL",         3)

# SALT_PEPPER_AMOUNT é uma tupla — mantida como constante (não tunável via env var simples)
SALT_PEPPER_AMOUNT  = (0.002, 0.015)

# Distorções geométricas (curvatura e desalinhamento da página)
ELASTIC_PROB        = _env("KD_ELASTIC_PROB",       0.4)
ELASTIC_ALPHA       = _env("KD_ELASTIC_ALPHA",      1)
ELASTIC_SIGMA       = _env("KD_ELASTIC_SIGMA",      50)
GRID_PROB           = _env("KD_GRID_PROB",           0.3)
GRID_DISTORT_LIMIT  = _env("KD_GRID_DISTORT_LIMIT", 0.15)
ROTATE_PROB         = _env("KD_ROTATE_PROB",         0.6)
ROTATE_LIMIT        = _env("KD_ROTATE_LIMIT",        3)
