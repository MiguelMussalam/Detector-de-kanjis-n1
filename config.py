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
WEIGHTS_DIR     = os.path.join(ROOT_DIR, "weights")

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
    "Yuji-Boku-Regular":        "https://raw.githubusercontent.com/Kinutafontfactory/Yuji/master/fonts/ttf/YujiBoku-Regular.ttf",
    "Zen-Antique-Regular":      "https://raw.githubusercontent.com/googlefonts/zen-antique/main/fonts/ttf/ZenAntique-Regular.ttf",
    "Kaisei-Tokumin-W5":        "https://raw.githubusercontent.com/Font-Kai/Kaisei-Tokumin/master/Fonts/ttf/FK-Kaisei-tokuminW5.ttf",
    "Hachi-Maru-Pop-Regular":   "https://raw.githubusercontent.com/noriokanisawa/HachiMaruPop/master/HachiMaruPop-Regular.ttf",
    "Stick-Regular":            "https://raw.githubusercontent.com/fontworks-fonts/Stick/master/fonts/ttf/Stick-Regular.ttf",
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

# ---------------------------------------------------------------------------
# Classificador — geração de crops sintéticos
# ---------------------------------------------------------------------------

# Diretórios do dataset gerado
CLASSIFIER_DATA_DIR   = os.path.join(DATA_DIR, "classifier")
CLASSIFIER_TRAIN_DIR  = os.path.join(CLASSIFIER_DATA_DIR, "train")
CLASSIFIER_VAL_DIR    = os.path.join(CLASSIFIER_DATA_DIR, "val")
CLASSIFIER_SANITY_DIR = os.path.join(CLASSIFIER_DATA_DIR, "sanity")

# Escopo do classificador
CLASSIFIER_KANJI_LEVEL   = _env("KD_CLF_LEVEL", "n1")
CLASSIFIER_INPUT_SIZE    = _env("KD_CLF_INPUT_SIZE", 64)

# Número de crops por classe (split 80/20)
CLASSIFIER_SAMPLES_TRAIN = _env("KD_CLF_SAMPLES_TRAIN", 40)
CLASSIFIER_SAMPLES_VAL   = _env("KD_CLF_SAMPLES_VAL", 10)

# Classe "outros" (rejeição de não-N1: hiragana/katakana, kanji fora do N1,
# letras latinas/dígitos — ex: onomatopeia tipo "RRRR" — e ruído/fundo vazio)
CLF_OUTROS_LABEL          = "OUTROS"
CLF_OUTROS_SAMPLES_TRAIN  = _env("KD_CLF_OUTROS_SAMPLES_TRAIN", 2000)
CLF_OUTROS_SAMPLES_VAL    = _env("KD_CLF_OUTROS_SAMPLES_VAL", 500)

# Proporção de cada subcategoria dentro do pool "outros" (soma ~1.0)
CLF_OUTROS_PROP_KANJI = _env("KD_CLF_OUTROS_PROP_KANJI", 0.35)  # kanji fora do N1
CLF_OUTROS_PROP_KANA  = _env("KD_CLF_OUTROS_PROP_KANA",  0.25)  # hiragana/katakana
CLF_OUTROS_PROP_LATIM = _env("KD_CLF_OUTROS_PROP_LATIM", 0.25)  # letras/dígitos latinos
CLF_OUTROS_PROP_RUIDO = _env("KD_CLF_OUTROS_PROP_RUIDO", 0.15)  # fundo vazio/ruído

# Sanity check visual
CLASSIFIER_SANITY_COUNT  = _env("KD_CLF_SANITY_COUNT", 20)

# Reprodutibilidade
CLASSIFIER_SEED_TRAIN    = _env("KD_CLF_SEED_TRAIN", 42)
CLASSIFIER_SEED_VAL      = _env("KD_CLF_SEED_VAL", 1337)

# Tamanhos de fonte simulados (multi-scale)
# Distribuição enviesada pra menores — em manga real kanji pequeno é mais comum
CLASSIFIER_FONT_SIZES    = [16, 20, 28, 40, 60, 96]

# Margem em volta do glyph antes do downsample para 64x64
CLASSIFIER_CANVAS_MARGIN = _env("KD_CLF_CANVAS_MARGIN", 0.15)

# Fundo do crop (papel — não branco puro)
CLF_BG_VALUE_MIN     = _env("KD_CLF_BG_MIN",     245)
CLF_BG_VALUE_MAX     = _env("KD_CLF_BG_MAX",     255)
CLF_BG_NOISE_STD     = _env("KD_CLF_BG_NOISE_STD", 2.0)

# Degradações
CLF_TRANSLATE_PROB   = _env("KD_CLF_TRANSLATE_PROB",   0.7)
CLF_TRANSLATE_MAX    = _env("KD_CLF_TRANSLATE_MAX",    0.10)

CLF_BLUR_PROB        = _env("KD_CLF_BLUR_PROB",        0.5)
CLF_BLUR_SIGMA_MIN   = _env("KD_CLF_BLUR_SIGMA_MIN",   0.3)
CLF_BLUR_SIGMA_MAX   = _env("KD_CLF_BLUR_SIGMA_MAX",   1.0)

CLF_NOISE_PROB       = _env("KD_CLF_NOISE_PROB",       0.5)
CLF_NOISE_STD_MIN    = _env("KD_CLF_NOISE_STD_MIN",    0.01)
CLF_NOISE_STD_MAX    = _env("KD_CLF_NOISE_STD_MAX",    0.05)

CLF_BRIGHTNESS_PROB  = _env("KD_CLF_BRIGHTNESS_PROB",  0.5)
CLF_BRIGHTNESS_RANGE = _env("KD_CLF_BRIGHTNESS_RANGE", 0.20)

CLF_CONTRAST_PROB    = _env("KD_CLF_CONTRAST_PROB",    0.5)
CLF_CONTRAST_RANGE   = _env("KD_CLF_CONTRAST_RANGE",   0.20)

CLF_MORFO_PROB       = _env("KD_CLF_MORFO_PROB",       0.3)
CLF_MORFO_K_MIN      = _env("KD_CLF_MORFO_K_MIN",      2)
CLF_MORFO_K_MAX      = _env("KD_CLF_MORFO_K_MAX",      3)

CLF_JPEG_PROB        = _env("KD_CLF_JPEG_PROB",        0.4)
CLF_JPEG_QUALITY_MIN = _env("KD_CLF_JPEG_QUALITY_MIN", 30)
CLF_JPEG_QUALITY_MAX = _env("KD_CLF_JPEG_QUALITY_MAX", 90)

CLF_ROTATION_PROB    = _env("KD_CLF_ROTATION_PROB",    0.3)
CLF_ROTATION_MAX     = _env("KD_CLF_ROTATION_MAX",     3.0)

# ---------------------------------------------------------------------------
# Classificador — modelo e treino
# ---------------------------------------------------------------------------

# Backbone (arquitetura da rede)
CLF_MODEL_ARCH   = _env("KD_CLF_MODEL_ARCH", "resnet18")
CLF_PRETRAINED   = _env("KD_CLF_PRETRAINED", True)

# Normalização (padrão ImageNet — compatível com backbones pré-treinados)
CLF_NORM_MEAN    = [0.485, 0.456, 0.406]
CLF_NORM_STD     = [0.229, 0.224, 0.225]

# DataLoader
CLF_BATCH_SIZE   = _env("KD_CLF_BATCH_SIZE", 128)
CLF_NUM_WORKERS  = _env("KD_CLF_NUM_WORKERS", 4)

# ---------------------------------------------------------------------------
# Classificador — loop de treino
# ---------------------------------------------------------------------------

CLF_EPOCHS         = _env("KD_CLF_EPOCHS", 30)
CLF_LR             = _env("KD_CLF_LR", 3e-4)
CLF_WEIGHT_DECAY   = _env("KD_CLF_WEIGHT_DECAY", 1e-4)
CLF_PATIENCE       = _env("KD_CLF_PATIENCE", 7)      # early stopping
CLF_PROJECT_NAME   = _env("KD_CLF_PROJECT_NAME", "kanji_classifier")

# ---------------------------------------------------------------------------
# Pipeline de inferência (detector + classificador)
# ---------------------------------------------------------------------------

# Pesos treinados (o do detector já usa weights/best.pt em outros lugares do repo)
DETECTOR_WEIGHTS_PATH = _env("KD_DETECTOR_WEIGHTS_PATH", os.path.join(WEIGHTS_DIR, "best.pt"))
CLF_WEIGHTS_PATH       = _env("KD_CLF_WEIGHTS_PATH", os.path.join(WEIGHTS_DIR, "classifier_best.pt"))

# Convenções decididas no roteiro do projeto (ver docs/pipeline.md)
PIPELINE_MIN_BBOX_HEIGHT = _env("KD_PIPELINE_MIN_BBOX_HEIGHT", 15)    # px — bbox menor que isso é descartada
PIPELINE_BBOX_PADDING    = _env("KD_PIPELINE_BBOX_PADDING", 0.10)     # expande bbox +10% antes do crop

# Inferência do detector (calibrados na rodada anterior — ver notebooks/01_detector_train.ipynb)
PIPELINE_DET_CONF  = _env("KD_PIPELINE_DET_CONF", 0.30)
PIPELINE_DET_IOU   = _env("KD_PIPELINE_DET_IOU", 0.40)
PIPELINE_DET_MAX_DET = _env("KD_PIPELINE_DET_MAX_DET", 1000)

# Abaixo desse valor, a predição do classificador é marcada como incerta (não é rejeição/descarte)
PIPELINE_CLS_CONF_LOW = _env("KD_PIPELINE_CLS_CONF_LOW", 0.5)

# ---------------------------------------------------------------------------
# ETL9 (validação fora de domínio — manuscrito, proxy de generalização)
# ---------------------------------------------------------------------------

# Diretório onde ficam os dados brutos do ETL9 (ver src/helper/etl9.py para a
# estrutura de pastas exata esperada pela biblioteca etl_data_reader — não é
# tão simples quanto jogar os arquivos soltos aqui dentro).
ETL9_DIR         = _env("KD_ETL9_DIR", os.path.join(DATA_DIR, "etl9"))

# Qual versão usar: "ETL9B" (binário, mais leve, 5 arquivos) ou "ETL9G" (grayscale, mais fiel, 50 arquivos)
ETL9_VERSION     = _env("KD_ETL9_VERSION", "ETL9G")
