import os

# Caminhos base
ROOT_DIR        = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR      = os.path.join(ROOT_DIR, "assets")
DATA_DIR        = os.path.join(ROOT_DIR, "data")

# Fontes
FONTS_DIR       = os.path.join(ASSETS_DIR, "fonts")

# Manga109
MANGA109_DIR    = os.path.join(DATA_DIR, "raw", "Manga109")
MANGA109_IMAGES = os.path.join(MANGA109_DIR, "images")

# Dataset sintético
SYNTHETIC_DIR   = os.path.join(DATA_DIR, "synthetic")
PAGES_DIR       = os.path.join(SYNTHETIC_DIR, "pages")

# Dataset YOLO (estrutura images/labels train/val)
DATASET_DIR     = os.path.join(DATA_DIR, "dataset")
TRAIN_IMG_DIR   = os.path.join(DATASET_DIR, "images", "train")
VAL_IMG_DIR     = os.path.join(DATASET_DIR, "images", "val")
TRAIN_LBL_DIR   = os.path.join(DATASET_DIR, "labels", "train")
VAL_LBL_DIR     = os.path.join(DATASET_DIR, "labels", "val")
VAL_SPLIT       = 0.1          # 10 % das páginas vão para validação

# URLs externas
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

# Parâmetros de geração
CROP_SIZE       = 640
PAGES_AMOUNT    = 5000         # páginas sintéticas para o dataset de treino
GAP_CHAR        = 4
GAP_COL         = 8

# Treino YOLO
YOLO_MODEL      = "yolo26n.pt"
EPOCHS          = 50
IMGSZ           = 640
BATCH           = 16
KAGGLE_WORKERS  = 2            # T4/P100 têm 2 vCPUs
LOCAL_WORKERS   = 4
PROJECT_NAME    = "kanji_detector"
KAGGLE_DATASET  = "kanji-detector-dataset"  # nome do dataset no Kaggle

# Restrição de região e posicionamento de texto
LIMITE_DESVIO_REGIAO = 25  # desvio padrão máximo aceitável para posicionamento de texto
MAX_TENTATIVAS_POSICAO = 10

# Degradações 2D — ordem de aplicação: geometria → tinta → scanner
# Morfologia (simulação de tinta)
EROSAO_PROB         = 0.3   # probabilidade de erosão por imagem
DILATACAO_PROB      = 0.3   # probabilidade de dilatação por imagem
MORFO_KERNEL        = 2     # tamanho do kernel NxN (2 ou 3)
MORFO_ITERACOES     = 1     # iterações de erosão/dilatação

# Ruído de scanner
GAUSS_NOISE_PROB    = 0.5   # probabilidade de ruído gaussiano
GAUSS_NOISE_STD_MIN = 0.05  # desvio mínimo (normalizado 0–1)
GAUSS_NOISE_STD_MAX = 0.15  # desvio máximo (normalizado 0–1)
SALT_PEPPER_PROB    = 0.4   # probabilidade de ruído sal e pimenta
SALT_PEPPER_AMOUNT  = (0.002, 0.015)  # fração de pixels afetados
BLUR_PROB           = 0.35  # probabilidade de desfoque gaussiano
BLUR_KERNEL         = 3     # tamanho do kernel (deve ser ímpar)

# Distorções geométricas (curvatura e desalinhamento da página)
ELASTIC_PROB        = 0.4   # probabilidade de distorção elástica
ELASTIC_ALPHA       = 1     # intensidade do deslocamento
ELASTIC_SIGMA       = 50    # suavidade do deslocamento (maior = mais suave)
GRID_PROB           = 0.3   # probabilidade de distorção de grade
GRID_DISTORT_LIMIT  = 0.15  # limite de distorção da grade
ROTATE_PROB         = 0.6   # probabilidade de rotação microscópica
ROTATE_LIMIT        = 3     # graus máximos (±)
