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
# letras latinas/dígitos — ex: onomatopeia e ruído/fundo vazio)
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

# Tamanhos de fonte simulados (multi-scale), com pesos de amostragem.
# Calibrado por medição empírica (ver src/helper/measure_real_stats.py): bbox
# real de caractere em manga tem mediana ~16-17px e p95 ~29-31px (10730 caixas,
# 60 páginas do Manga109 via nosso detector). Os tamanhos antigos (40/60/96)
# representavam detalhe que praticamente nunca ocorre em produção — tirados.
CLASSIFIER_FONT_SIZES        = [12, 14, 16, 18, 22, 28, 36]
CLASSIFIER_FONT_SIZE_WEIGHTS = [8, 16, 22, 20, 16, 12, 6]

# Margem em volta do glyph antes do downsample para 64x64
CLASSIFIER_CANVAS_MARGIN = _env("KD_CLF_CANVAS_MARGIN", 0.15)

# Fundo do crop (papel — não branco puro)
CLF_BG_VALUE_MIN     = _env("KD_CLF_BG_MIN",     245)
CLF_BG_VALUE_MAX     = _env("KD_CLF_BG_MAX",     255)
CLF_BG_NOISE_STD     = _env("KD_CLF_BG_NOISE_STD", 2.0)

# Fundo real (patches recortados de páginas do Manga109 sem nenhum glifo por
# perto — screentone, traço de arte, hachura). Preferido sobre o fundo
# sintético (papel liso) acima quando disponível; ver
# src/helper/harvest_backgrounds.py. Cai pro fundo sintético automaticamente
# se a pasta não existir (ex: ambiente Kaggle sem esse dataset anexado).
def _buscar_backgrounds_real_dir():
    """
    Retorna o diretório com os patches de fundo real (subpastas claro/escuro).
    No local, 'data/backgrounds_real'. No Kaggle, busca dinamicamente uma
    pasta anexada como dataset contendo essas duas subpastas -- harvesting
    roda uma vez localmente (precisa do Manga109 + detector) e o resultado
    (só imagens pequenas) é reaproveitado via upload, sem precisar reanexar
    o Manga109 inteiro no notebook do classificador.
    """
    caminho_local = os.path.join(DATA_DIR, "backgrounds_real")
    if os.path.isdir(os.path.join(caminho_local, "claro")) and os.path.isdir(os.path.join(caminho_local, "escuro")):
        return caminho_local

    kaggle_input = "/kaggle/input"
    if os.path.exists(kaggle_input):
        for root, dirs, _ in os.walk(kaggle_input):
            if "claro" in dirs and "escuro" in dirs:
                print(f"[INFO] backgrounds_real encontrado no Kaggle: {root}")
                return root
    return caminho_local

BACKGROUNDS_REAL_DIR   = _buscar_backgrounds_real_dir()
CLF_BG_PATCH_SIZE      = _env("KD_CLF_BG_PATCH_SIZE", 72)   # um pouco maior que o input, pra permitir crop aleatorio variado
CLF_BG_HARVEST_COUNT   = _env("KD_CLF_BG_HARVEST_COUNT", 4000)
CLF_BG_HARVEST_PAGINAS = _env("KD_CLF_BG_HARVEST_PAGINAS", 800)  # teto de paginas a varrer (fundo escuro e raro, precisa de mais paginas pra encher a cota)
# Patch precisa estar perto de algum glifo detectado (nao so longe de todos) --
# senao a amostragem cai em qualquer lugar da pagina (arte do painel, rosto,
# cabelo), que costuma ser bem mais "carregado" visualmente do que o fundo
# que realmente fica atras de texto (quase sempre dentro de um balao, quase
# liso). Limite = multiplo do patch_size.
CLF_BG_MAX_DIST_FACTOR = _env("KD_CLF_BG_MAX_DIST_FACTOR", 4.0)

# Filtro de "quietude": so aceita o patch se o desvio-padrao dos pixels for
# baixo -- reduz a variancia geral do pool (menos arte carregada, mais fundo
# de fato liso/uniforme atras do texto). Esse filtro pega tanto fundo BRANCO
# quanto fundo PRETO uniforme (alguns paineis de manga sao preto solido com
# texto branco), ja que os dois tem desvio baixo -- so a media que difere.
CLF_BG_MAX_STD = _env("KD_CLF_BG_MAX_STD", 25.0)

# Separa o pool em "claro" e "escuro" (media de pixel abaixo do limite = escuro).
# Fundo escuro é raro no corpus (~poucos %), mas existe (paineis pretos com
# texto branco pra dar impacto dramatico) e precisa de pool proprio: exige
# inverter a polaridade do glifo (traço branco, não preto) na composição --
# ver CLF_BG_ESCURO_PROB em generate_crops.py.
CLF_BG_DARK_MEAN_THRESHOLD = _env("KD_CLF_BG_DARK_MEAN_THRESHOLD", 80.0)
CLF_BG_HARVEST_COUNT_ESCURO = _env("KD_CLF_BG_HARVEST_COUNT_ESCURO", 300)
CLF_BG_ESCURO_PROB = _env("KD_CLF_BG_ESCURO_PROB", 0.08)  # proporcao usada na geracao, nao a natural

# Garantia de legibilidade: depois de aplicar fundo real + blur + ruido, o
# glifo precisa continuar visivelmente diferente do fundo ao redor (senao a
# degradacao "esvazia" o caractere e a amostra vira ruido com rotulo errado).
# Se o contraste cair abaixo do limite, tenta de novo (novo sorteio de
# fundo/blur/ruido) ate CLF_MAX_TENTATIVAS vezes; se ainda assim falhar,
# gera uma versao sem blur/ruido pra garantir a legibilidade.
CLF_MIN_CONTRASTE   = _env("KD_CLF_MIN_CONTRASTE", 70.0)
CLF_MAX_TENTATIVAS  = _env("KD_CLF_MAX_TENTATIVAS", 6)

# Antes mesmo de degradar: algumas fontes rendem em branco pra um caractere
# raro (glifo ausente/quebrado naquele tamanho) -- sem checagem, isso vira
# uma amostra 100% vazia que passa direto (nao tem glifo pra medir contraste
# contra). Se a fonte sorteada render menos que isso de pixel escuro, troca
# de fonte antes de seguir pro resto do pipeline.
CLF_MIN_PIXELS_GLIFO = _env("KD_CLF_MIN_PIXELS_GLIFO", 250)

# Degradações
CLF_TRANSLATE_PROB   = _env("KD_CLF_TRANSLATE_PROB",   0.7)
CLF_TRANSLATE_MAX    = _env("KD_CLF_TRANSLATE_MAX",    0.10)

# Blur e ruído calibrados juntos contra a nitidez real medida (variância do
# Laplaciano, ver src/helper/measure_real_stats.py: mediana real ~210, p95
# ~940). As duas degradações não são independentes nessa métrica -- ruído
# pixel-a-pixel infla a variância do Laplaciano quase tanto quanto falta de
# blur, entao foram achadas em busca conjunta (grid search local), não isoladas.
CLF_BLUR_PROB        = _env("KD_CLF_BLUR_PROB",        1.0)
CLF_BLUR_SIGMA_MIN   = _env("KD_CLF_BLUR_SIGMA_MIN",   0.9)
CLF_BLUR_SIGMA_MAX   = _env("KD_CLF_BLUR_SIGMA_MAX",   1.5)

CLF_NOISE_PROB       = _env("KD_CLF_NOISE_PROB",       0.7)
CLF_NOISE_STD_MIN    = _env("KD_CLF_NOISE_STD_MIN",    0.01)
CLF_NOISE_STD_MAX    = _env("KD_CLF_NOISE_STD_MAX",    0.03)

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

# Fine-tuning a partir de um checkpoint ja treinado (dado real + sintetico
# misturado). Vazio ("") = treina do zero (pesos ImageNet), como antes.
CLF_FINETUNE_FROM = _env("KD_CLF_FINETUNE_FROM", "")
CLF_FINETUNE_LR   = _env("KD_CLF_FINETUNE_LR", 2e-5)   # bem mais baixo que CLF_LR

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

# ---------------------------------------------------------------------------
# Alinhamento Manga109 (dado real para fine-tuning do classificador)
# ---------------------------------------------------------------------------
# Deduz o rótulo de cada caractere real usando a transcrição das falas do
# Manga109 (<text>) + as bboxes que o NOSSO detector encontra na página inteira.
# Ver src/helper/manga109_align.py.

MANGA109_ALIGN_DATA_DIR   = os.path.join(DATA_DIR, "classifier_real")
MANGA109_ALIGN_TRAIN_DIR  = os.path.join(MANGA109_ALIGN_DATA_DIR, "train")
MANGA109_ALIGN_VAL_DIR    = os.path.join(MANGA109_ALIGN_DATA_DIR, "val")

# Split por volume (não por página/crop, para não vazar dado de val no treino).
# Mesmo conjunto de 10 volumes historicamente usado como val do Manga109
# (Ogawa et al., 2018), com KarakuriDouji trocado por KarappoHighschool —
# não existe nessa versão do dataset baixada.
MANGA109_ALIGN_VAL_VOLUMES = [
    "YamatoNoHane", "RisingGirl", "Hamlet", "TaiyouNiSmash",
    "UchuKigekiM774", "WarewareHaOniDearu", "YumeNoKayoiji",
    "KarappoHighschool", "EverydayOsakanaChan", "HealingPlanet",
]

# Tolerância adaptativa (fração da largura mediana das boxes da linha) para
# agrupar caixas do detector na mesma coluna de leitura — substituiu um valor
# fixo em pixel, que se mostrou frágil em painéis com colunas compactas.
MANGA109_ALIGN_COLUNA_TOL_FRAC = _env("KD_ALIGN_COLUNA_TOL_FRAC", 0.5)

# Linhas com mais colunas que isso são descartadas (risco maior de
# agrupamento errado) mesmo que a contagem de caracteres bata.
MANGA109_ALIGN_MAX_COLUNAS = _env("KD_ALIGN_MAX_COLUNAS", 3)

# Pontuação/símbolos removidos da transcrição antes de comparar contagem
# (o detector não foi treinado para caixar esses símbolos como "glifo")
MANGA109_ALIGN_PONTUACAO = "。、！？!?…‼♥「」『』・（）() 　.,"
