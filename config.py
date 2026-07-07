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
def _buscar_manga109_diretorio():
    """
    Retorna o diretório base do Manga109.
    No local, será 'data/raw/Manga109'.
    No Kaggle, busca dinamicamente uma pasta contendo 'images' e 'annotations'
    ou cujo nome contenha 'manga109'.
    """
    caminho_local = os.path.join(DATA_DIR, "raw", "Manga109")
    if os.path.exists(os.path.join(caminho_local, "images")) and os.path.exists(os.path.join(caminho_local, "annotations")):
        return caminho_local

    # Busca dinâmica no ambiente Kaggle
    kaggle_input = "/kaggle/input"
    if os.path.exists(kaggle_input):
        for root, dirs, _ in os.walk(kaggle_input):
            if "images" in dirs and "annotations" in dirs:
                print(f"[INFO] Manga109 dir encontrado no Kaggle: {root}")
                return root
            if "manga109" in root.lower() and ("images" in dirs or "annotations" in dirs):
                print(f"[INFO] Manga109 dir parcial encontrado no Kaggle: {root}")
                return root
    return caminho_local

MANGA109_DIR = _buscar_manga109_diretorio()
MANGA109_IMAGES = os.path.join(MANGA109_DIR, "images")
MANGA109_ANNOTATIONS = os.path.join(MANGA109_DIR, "annotations")

# Caminho do data.yaml do dataset YOLO (gerado pelo notebook de treino)
DATA_YAML = _env("KD_DATA_YAML", os.path.join(DATA_DIR, "data.yaml"))

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

# Treino YOLO
YOLO_MODEL      = _env("KD_YOLO_MODEL",  "yolo26n.pt")
EPOCHS          = _env("KD_EPOCHS",      50)
IMGSZ           = _env("KD_IMGSZ",       640)
BATCH           = _env("KD_BATCH",       16)
KAGGLE_WORKERS  = _env("KD_KAGGLE_WORKERS", 2)   # T4/P100 têm 2 vCPUs
LOCAL_WORKERS   = _env("KD_LOCAL_WORKERS",  4)
PROJECT_NAME    = _env("KD_PROJECT_NAME", "kanji_detector")
