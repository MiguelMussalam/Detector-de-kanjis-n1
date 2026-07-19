import os


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


ROOT_DIR        = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR      = os.path.join(ROOT_DIR, "assets")
DATA_DIR        = os.path.join(ROOT_DIR, "data")
WEIGHTS_DIR     = os.path.join(ROOT_DIR, "weights")
FONTS_DIR       = os.path.join(ASSETS_DIR, "fonts")


def _buscar_manga109_diretorio():
    """
    Retorna o diretório base do Manga109.
    No local, será 'data/raw/Manga109'.
    No Kaggle, varre TODA a árvore de /kaggle/input em busca de uma pasta com
    'images' e 'annotations' juntas (match forte) -- só cai pro match fraco
    (nome contendo 'manga109' + só uma das duas pastas) se nenhum match forte
    existir em lugar nenhum.

    Precisa varrer a árvore INTEIRA antes de decidir (não retornar no primeiro
    achado): datasets Kaggle não relacionados podem ter "manga109" no nome
    (ex: dataset Roboflow "manga109-character-bouding-box", que tem uma pasta
    "images" no formato YOLO sem nenhuma relação com o corpus completo) e
    bater no match fraco por acidente antes da busca alcançar o dataset certo
    -- bug real desta sessão: o Roboflow "test/images" foi escolhido no lugar
    do corpus completo (que também estava anexado), gerando 0 páginas
    sintéticas silenciosamente (generate_pages.py não tinha nenhum XML pra
    achar em MANGA109_ANNOTATIONS, mas seguiu em frente sem erro).
    """
    caminho_local = os.path.join(DATA_DIR, "raw", "Manga109")
    if os.path.exists(os.path.join(caminho_local, "images")) and os.path.exists(os.path.join(caminho_local, "annotations")):
        return caminho_local

    kaggle_input = "/kaggle/input"
    if os.path.exists(kaggle_input):
        match_parcial = None
        for root, dirs, _ in os.walk(kaggle_input):
            if "images" in dirs and "annotations" in dirs:
                print(f"[INFO] Manga109 dir encontrado no Kaggle: {root}")
                return root
            if match_parcial is None and "manga109" in root.lower() and ("images" in dirs or "annotations" in dirs):
                match_parcial = root
        if match_parcial is not None:
            print(f"[AVISO] Nenhuma pasta com 'images' e 'annotations' juntas encontrada -- "
                  f"usando match fraco (nome parcial, pode ser o dataset errado): {match_parcial}")
            return match_parcial
    return caminho_local

MANGA109_DIR = _buscar_manga109_diretorio()
MANGA109_IMAGES = os.path.join(MANGA109_DIR, "images")
MANGA109_ANNOTATIONS = os.path.join(MANGA109_DIR, "annotations")

# Caminho do data.yaml do dataset YOLO (gerado pelo notebook de treino)
DATA_YAML = _env("KD_DATA_YAML", os.path.join(DATA_DIR, "data.yaml"))

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

YOLO_MODEL      = _env("KD_YOLO_MODEL",  "yolo26n.pt")
EPOCHS          = _env("KD_EPOCHS",      50)
IMGSZ           = _env("KD_IMGSZ",       640)
BATCH           = _env("KD_BATCH",       16)
KAGGLE_WORKERS  = _env("KD_KAGGLE_WORKERS", 2)   # T4/P100 têm 2 vCPUs
LOCAL_WORKERS   = _env("KD_LOCAL_WORKERS",  4)
PROJECT_NAME    = _env("KD_PROJECT_NAME", "kanji_detector")

CLASSIFIER_DATA_DIR   = os.path.join(DATA_DIR, "classifier")
CLASSIFIER_TRAIN_DIR  = os.path.join(CLASSIFIER_DATA_DIR, "train")
CLASSIFIER_VAL_DIR    = os.path.join(CLASSIFIER_DATA_DIR, "val")
CLASSIFIER_SANITY_DIR = os.path.join(CLASSIFIER_DATA_DIR, "sanity")

CLASSIFIER_KANJI_LEVEL   = _env("KD_CLF_LEVEL", "n1")
CLASSIFIER_INPUT_SIZE    = _env("KD_CLF_INPUT_SIZE", 64)

CLASSIFIER_SAMPLES_TRAIN = _env("KD_CLF_SAMPLES_TRAIN", 200)
CLASSIFIER_SAMPLES_VAL   = _env("KD_CLF_SAMPLES_VAL", 40)

CLF_OUTROS_LABEL          = "OUTROS"
# Rodada 100% sintetica (sem merge de dado real) -- ver conversa: o treino
# anterior misturou ~91% real no OUTROS contra 0% real no N1, e o modelo
# aprendeu a distinguir dominio (real vs sintetico) em vez de conteudo. Essa
# rodada isola a pergunta "a geracao sintetica melhorada ja basta sozinha?"
# removendo o dado real inteiramente -- por isso o volume sintetico de OUTROS
# subiu pra ficar numa escala parecida com o que tinhamos antes (era 2000/500
# sintetico + ate 20000/5000 real).
CLF_OUTROS_SAMPLES_TRAIN  = _env("KD_CLF_OUTROS_SAMPLES_TRAIN", 20000)
CLF_OUTROS_SAMPLES_VAL    = _env("KD_CLF_OUTROS_SAMPLES_VAL", 5000)

# Proporção de cada subcategoria dentro do pool "outros" (soma ~1.0)
CLF_OUTROS_PROP_KANJI = _env("KD_CLF_OUTROS_PROP_KANJI", 0.35)  # kanji fora do N1
CLF_OUTROS_PROP_KANA  = _env("KD_CLF_OUTROS_PROP_KANA",  0.25)  # hiragana/katakana
CLF_OUTROS_PROP_LATIM = _env("KD_CLF_OUTROS_PROP_LATIM", 0.25)  # letras/dígitos latinos
CLF_OUTROS_PROP_RUIDO = _env("KD_CLF_OUTROS_PROP_RUIDO", 0.15)  # fundo vazio/ruído

CLASSIFIER_SANITY_COUNT  = _env("KD_CLF_SANITY_COUNT", 20)

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

# Contraste (acima) mede só brilho médio -- fica cego pra dois jeitos de
# perder legibilidade que passam nele mesmo assim: traços que grudaram numa
# mancha sólida (fonte pesada + kanji com muitos traços + tamanho pequeno) ou
# que se desfizeram em fiapos esparsos (erosão/blur agressivo em traço já
# fino). "Fill ratio" mede a fração de pixel "tinta" dentro do retângulo do
# próprio glifo -- um kanji legível tem uma faixa saudável, nem quase 0%
# (erodido) nem perto de 100% (virou blob). Faixa calibrada empiricamente
# (ver src/classifier/generate_crops.py generate_sanity + CSV de instrumentação).
CLF_MIN_FILL_RATIO = _env("KD_CLF_MIN_FILL_RATIO", 0.12)
CLF_MAX_FILL_RATIO = _env("KD_CLF_MAX_FILL_RATIO", 0.75)

# Igual CLF_MAX_TENTATIVAS, mas pro render base (fonte/tamanho/morfologia/
# rotação/posição) -- essa parte só é sorteada uma vez hoje; se sair ruim
# (fora da faixa de fill ratio ou pixel insuficiente), precisa re-sortear
# ANTES de entrar no loop de fundo/blur, senão todo o resto do retry gasta
# tentativa em cima de um glifo que já nasceu ilegível.
CLF_MAX_TENTATIVAS_MORFO = _env("KD_CLF_MAX_TENTATIVAS_MORFO", 4)

# Fontes bold/pesadas/decorativas -- achadas na auditoria visual como causa
# do "vira mancha sólida": kanji com muitos traços grudam quando a fonte já é
# grossa por design, ainda mais se a morfologia "dilate" (engrossar) for
# sorteada em cima. Pra essas, só "erode" (afinar) é permitido.
CLF_FONTES_PESADAS = [
    "Dela-Gothic-One.ttf", "Reggae-One.ttf", "BIZ-UDPGothic-Bold.ttf",
    "Stick-Regular.ttf", "Klee-One-SemiBold.ttf",
]

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
# k=3 auditado (src/helper/filter_audit.py) e caiu no penhasco de acuracia
# especifico da erosao (92.5% em k=2 -> 40% em k=3, fill_ratio nao separa bem
# essa faixa) -- k_max travado em 2 (junto com k_min=2, kernel fica fixo em 2
# quando a morfologia dispara) mantem a amostra dentro do platô seguro.
CLF_MORFO_K_MAX      = _env("KD_CLF_MORFO_K_MAX",      2)

CLF_JPEG_PROB        = _env("KD_CLF_JPEG_PROB",        0.4)
CLF_JPEG_QUALITY_MIN = _env("KD_CLF_JPEG_QUALITY_MIN", 30)
CLF_JPEG_QUALITY_MAX = _env("KD_CLF_JPEG_QUALITY_MAX", 90)

CLF_ROTATION_PROB    = _env("KD_CLF_ROTATION_PROB",    0.3)
CLF_ROTATION_MAX     = _env("KD_CLF_ROTATION_MAX",     3.0)

CLF_MODEL_ARCH   = _env("KD_CLF_MODEL_ARCH", "resnet18")
CLF_PRETRAINED   = _env("KD_CLF_PRETRAINED", True)

# Normalização (padrão ImageNet — compatível com backbones pré-treinados)
CLF_NORM_MEAN    = [0.485, 0.456, 0.406]
CLF_NORM_STD     = [0.229, 0.224, 0.225]

CLF_BATCH_SIZE   = _env("KD_CLF_BATCH_SIZE", 128)
CLF_NUM_WORKERS  = _env("KD_CLF_NUM_WORKERS", 4)

CLF_EPOCHS         = _env("KD_CLF_EPOCHS", 30)
CLF_LR             = _env("KD_CLF_LR", 3e-4)
CLF_WEIGHT_DECAY   = _env("KD_CLF_WEIGHT_DECAY", 1e-4)
CLF_PATIENCE       = _env("KD_CLF_PATIENCE", 7)      # early stopping
CLF_PROJECT_NAME   = _env("KD_CLF_PROJECT_NAME", "kanji_classifier")

# Fine-tuning a partir de um checkpoint ja treinado (dado real + sintetico
# misturado). Vazio ("") = treina do zero (pesos ImageNet), como antes.
CLF_FINETUNE_FROM = _env("KD_CLF_FINETUNE_FROM", "")
CLF_FINETUNE_LR   = _env("KD_CLF_FINETUNE_LR", 2e-5)   # bem mais baixo que CLF_LR

DETECTOR_WEIGHTS_PATH = _env("KD_DETECTOR_WEIGHTS_PATH", os.path.join(WEIGHTS_DIR, "best.pt"))
CLF_WEIGHTS_PATH       = _env("KD_CLF_WEIGHTS_PATH", os.path.join(WEIGHTS_DIR, "classifier_best.pt"))

PIPELINE_MIN_BBOX_HEIGHT = _env("KD_PIPELINE_MIN_BBOX_HEIGHT", 15)    # px — bbox menor que isso é descartada
PIPELINE_BBOX_PADDING    = _env("KD_PIPELINE_BBOX_PADDING", 0.10)     # expande bbox +10% antes do crop

# Calibrados empiricamente em notebooks/01_detector_train.ipynb
PIPELINE_DET_CONF  = _env("KD_PIPELINE_DET_CONF", 0.30)
PIPELINE_DET_IOU   = _env("KD_PIPELINE_DET_IOU", 0.40)
PIPELINE_DET_MAX_DET = _env("KD_PIPELINE_DET_MAX_DET", 1000)

# Abaixo desse valor, a predição do classificador é marcada como incerta (não é rejeição/descarte)
PIPELINE_CLS_CONF_LOW = _env("KD_PIPELINE_CLS_CONF_LOW", 0.5)

# ETL9: validação fora de domínio (manuscrito, proxy de generalização).
# Diretório onde ficam os dados brutos do ETL9 (ver src/helper/etl9.py para a
# estrutura de pastas exata esperada pela biblioteca etl_data_reader — não é
# tão simples quanto jogar os arquivos soltos aqui dentro).
ETL9_DIR         = _env("KD_ETL9_DIR", os.path.join(DATA_DIR, "etl9"))

# Qual versão usar: "ETL9B" (binário, mais leve, 5 arquivos) ou "ETL9G" (grayscale, mais fiel, 50 arquivos)
ETL9_VERSION     = _env("KD_ETL9_VERSION", "ETL9G")

# Alinhamento Manga109: gera negativos reais pra classe OUTROS (não N1, ver
# src/helper/manga109_align.py), deduzindo o rótulo de cada caractere real
# a partir da transcrição das falas do Manga109 (<text>) cruzada com as
# bboxes que o NOSSO detector encontra na página inteira.
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

# Duas checagens extras (ver colunas_sao_confiaveis em manga109_align.py),
# pegam erro de alinhamento que passa pela contagem batendo (caractere
# perdido pelo detector + caixa espuria em outro lugar da linha cancelam o
# total, mas o zip bbox<->char sai deslocado a partir do erro):
#   1. Coluna isolada de 1 caixa entre colunas bem maiores -- normalmente e
#      um artefato (pontuacao/ruido que o detector caixou por engano).
#   2. Salto de Y muito maior que a altura tipica das boxes da coluna --
#      sinaliza um caractere que o detector nao achou naquele ponto.
MANGA109_ALIGN_COLUNA_ISOLADA_MIN_VIZINHA = _env("KD_ALIGN_COLUNA_ISOLADA_MIN_VIZINHA", 3)
MANGA109_ALIGN_GAP_Y_FATOR = _env("KD_ALIGN_GAP_Y_FATOR", 1.6)

# Pontuação/símbolos removidos da transcrição antes de comparar contagem
# (o detector não foi treinado para caixar esses símbolos como "glifo")
MANGA109_ALIGN_PONTUACAO = "。、！？!?…‼♥「」『』・（）() 　.,"

# Geracao sintetica de paginas pro detector (src/detector/synth_page.py,
# src/detector/generate_pages.py): ancora glifos sinteticos nas bboxes de
# linha <text> oficiais do Manga109 (em vez de posicao arbitraria, como o
# gerador antigo deletado no commit 8292b18 fazia -- aquele arriscava deixar
# texto real nao rotulado em outro canto do mesmo recorte).
DETSYN_OUTPUT_DIR = os.path.join(DATA_DIR, "detector_synth")

# Quantas paginas-fonte processar por rodada de geracao.
DETSYN_N_PAGES = _env("KD_DETSYN_N_PAGES", 300)

# Fracao de celulas que mantem o caractere original da transcricao (preserva
# padrao realista de repeticao/densidade); o resto vira kanji N1 aleatorio
# (expoe o detector a variedade das 1232 classes, nao so ao que apareceu nas
# ~17 paginas reais anotadas). Chute educado inicial, sem calibracao contra
# dado real (nao existe "proporcao correta" conhecida) -- ajustavel depois
# via benchmark.
DETSYN_PROP_ORIGINAL_CHAR = _env("KD_DETSYN_PROP_ORIGINAL_CHAR", 0.3)

# Fracao das paginas-fonte sorteadas que sao paginas SEM nenhum <text> (arte
# pura) em vez de paginas com linhas sintetizadas -- viram exemplo puramente
# negativo (label vazio), reforcando que arte/screentone/painel nao deve ser
# detectado. Nao apaga texto de pagina com <text> e deixa sem rotulo (isso
# ensinaria o detector a ignorar caractere de verdade) -- so escolhe pagina
# que ja nao tem nenhum texto anotado pra comecar.
DETSYN_PROB_PAGINA_SEM_TEXTO = _env("KD_DETSYN_PROB_PAGINA_SEM_TEXTO", 0.1)

# Faixa plausivel de tamanho de celula (lado no eixo de leitura, px) pra
# confiar na subdivisao geometrica de uma linha em N caracteres iguais.
# Calibrado com margem em torno do tamanho real medido (measure_real_stats.py:
# lado mediano ~16-17px, p95 ~29-31px). Fora da faixa, a linha e' tratada
# como provavel multi-coluna (Manga109 nao da informacao de onde cada coluna
# comeca) -- tinta apagada, sem tentar adivinhar o layout, sem rotulo.
DETSYN_CELL_MIN_PX = _env("KD_DETSYN_CELL_MIN_PX", 8.0)
DETSYN_CELL_MAX_PX = _env("KD_DETSYN_CELL_MAX_PX", 60.0)

# So usado no fallback de coluna/linha UNICA (grupos=1) de dividir_em_celulas
# (src/detector/synth_page.py): eixo curto nao pode ser maior que este
# multiplo do tamanho de celula do eixo longo -- acima disso, nem a estimativa
# de grade multi-coluna achou um grid plausivel, entao a geometria e tratada
# como degenerada de verdade (nao so "provavel multi-coluna", que a grade 2D
# ja cobre separadamente).
DETSYN_MAX_RAZAO_LARGURA = _env("KD_DETSYN_MAX_RAZAO_LARGURA", 2.2)

# Margem (px) alem da bbox da linha ao apagar a tinta original -- cobre
# antialiasing/traco que vaza um pouco pra fora da bbox anotada.
DETSYN_APAGAR_MARGEM_PX = _env("KD_DETSYN_APAGAR_MARGEM_PX", 2)

# Validação full-corpus (src/helper/manga109_corpus.py, src/helper/corpus_validate.py):
# piso de área de bbox por caractere (não largura/altura isolada -- robusto a
# linha multi-coluna, onde dividir por um só eixo subestimaria o tamanho real
# do caractere), parte do filtro automático de transcrição oficial corrompida
# que substitui a verificação visual manual (inviável em ~10.600 páginas).
# Calibrado com margem generosa abaixo da área real medida (ver
# measure_real_stats.py: lado mediano ~16-17px -> área mediana ~270-290).
# Só existe piso, não teto: um teto foi tentado e descartado após auditoria
# em 25 volumes (14/19 exclusões eram onomatopeia/título dramático em fonte
# gigante -- convenção normal de mangá, não sinal de transcrição corrompida).
CORPUS_CELL_AREA_MIN = _env("KD_CORPUS_CELL_AREA_MIN", 60.0)
