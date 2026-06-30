import os
import random
import glob
from PIL import Image, ImageDraw, ImageFont, ImageStat
from config import MANGA109_IMAGES, PAGES_DIR, CROP_SIZE, PAGES_AMOUNT, FONTS_DIR, LIMITE_DESVIO_REGIAO, MAX_TENTATIVAS_POSICAO, BBOX_MARGEM
from src.helper.chars import get_supported_kanjis_from_fonts
from src.helper.degradacoes import aplicar_degradacoes

pages = glob.glob(os.path.join(MANGA109_IMAGES, "**", "*.jpg"), recursive=True)
fonts = glob.glob(os.path.join(FONTS_DIR, "*.ttf"))
kanjis = get_supported_kanjis_from_fonts()

def get_croped_image():
    random_page = random.choice(pages)
    image = Image.open(random_page).convert("RGB")
    largura, altura = image.size
    if largura < CROP_SIZE or altura < CROP_SIZE:
        print(f"Imagem {random_page} é muito pequena para o tamanho de corte {CROP_SIZE}.")
        return get_croped_image()
    x = random.randint(0, largura - CROP_SIZE)
    y = random.randint(0, altura - CROP_SIZE)

    return image.crop((x, y, x + CROP_SIZE, y + CROP_SIZE))

def calcular_tamanho_bloco(fonte, n_colunas, chars_por_coluna, direcao, gap_char=4, gap_col=8):
    bbox   = fonte.getbbox("字")
    char_w = bbox[2] - bbox[0]
    char_h = bbox[3] - bbox[1]

    if direcao == "vertical":
        largura = n_colunas  * (char_w + gap_col)
        altura  = chars_por_coluna * (char_h + gap_char)
    else:
        largura = chars_por_coluna * (char_w + gap_char)
        altura  = n_colunas  * (char_h + gap_col)

    return largura, altura

def analisar_regiao(crop: Image.Image, paste_x: int, paste_y: int,
                    bloco_w: int, bloco_h: int) -> dict:
    """
    Analisa a região do crop onde o bloco de texto seria colado.
    Retorna desvio padrão (uniformidade) e brilho médio (para cor de contraste).
    """
    regiao = crop.crop((paste_x, paste_y,
                        paste_x + bloco_w,
                        paste_y + bloco_h))
    stat   = ImageStat.Stat(regiao.convert("L"))
    return {
        "desvio": stat.stddev[0],   # baixo = uniforme, alto = caótico
        "brilho": stat.mean[0],     # 0 = preto, 255 = branco
    }

def escolher_cor_tinta(brilho: float) -> tuple:
    """
    Escolhe a cor da tinta baseado no brilho médio do fundo local.
    Garante contraste entre texto e fundo.
    """
    if brilho > 180:
        # fundo claro (balão branco, área clara) → tinta escura
        v = random.randint(0, 40)
    elif brilho < 80:
        # fundo escuro → tinta clara
        v = random.randint(215, 255)
    else:
        # fundo intermediário → sorteia qual extremo contrasta mais
        v = random.choice([
            random.randint(0, 40),
            random.randint(215, 255)
        ])
    return (v, v, v)  # escala de cinza — kanjis são monocromáticos


def _expandir_bboxes(bboxes: list, margem: float = 0.10) -> list:
    """
    Expande cada bounding box em `margem` proporcional ao seu tamanho
    (ex: 0.10 = +10% em cada direção), clampeando ao limite do crop.
    Garante que o modelo aprenda a incluir as extremidades dos traços
    sem cortar as pontas dos kanjis.
    """
    resultado = []
    for x, y, w, h in bboxes:
        pad_x = w * margem
        pad_y = h * margem
        x_new = max(0, x - pad_x)
        y_new = max(0, y - pad_y)
        w_new = min(CROP_SIZE - x_new, w + 2 * pad_x)
        h_new = min(CROP_SIZE - y_new, h + 2 * pad_y)
        resultado.append((x_new, y_new, w_new, h_new))
    return resultado


def render_kanji(crop: Image.Image):
    fonte_path = random.choice(fonts)
    tam_fonte  = random.randint(24, 48)
    n_colunas  = random.randint(1, 4)
    chars_por_coluna = random.randint(4, 12)
    direcao    = random.choice(["vertical", "horizontal"])
    gap_char   = random.randint(2, 6)
    gap_col    = random.randint(4, 10)

    font = ImageFont.truetype(fonte_path, tam_fonte)

    bloco_w, bloco_h = calcular_tamanho_bloco(font, n_colunas, chars_por_coluna, direcao, gap_char, gap_col)

    max_x = CROP_SIZE - bloco_w
    max_y = CROP_SIZE - bloco_h

    if max_x <= 0 or max_y <= 0:
        return None

    posicao_encontrada = False
    for _ in range(MAX_TENTATIVAS_POSICAO):
        candidato_x = random.randint(0, max_x)
        candidato_y = random.randint(0, max_y)

        stats = analisar_regiao(crop, candidato_x, candidato_y, bloco_w, bloco_h)

        if stats["desvio"] <= LIMITE_DESVIO_REGIAO:
            inicio_x = candidato_x
            inicio_y = candidato_y
            cor_tinta = escolher_cor_tinta(stats["brilho"])
            posicao_encontrada = True
            break

    if not posicao_encontrada:
        return None  # nenhuma posição uniforme encontrada, descartar este crop

    # calcula a cor da borda/contorno baseada no contraste com a cor da tinta
    stroke_fill = (255, 255, 255) if cor_tinta[0] < 128 else (0, 0, 0)

    # gera caracteres para todas as colunas
    todas_colunas = []
    for _ in range(n_colunas):
        coluna = [random.choice(kanjis) for _ in range(chars_por_coluna)]
        todas_colunas.append(coluna)

    # calcula tamanho de um caractere de referência para o espaçamento do cursor
    bbox   = font.getbbox("字")
    char_w = bbox[2] - bbox[0]
    char_h = bbox[3] - bbox[1]

    draw   = ImageDraw.Draw(crop)
    bboxes = []

    if direcao == "vertical":
        cursor_x = inicio_x
        for coluna in todas_colunas:
            cursor_y = inicio_y
            for char in coluna:
                draw.text((cursor_x, cursor_y), char, font=font, fill=cor_tinta, stroke_width=2, stroke_fill=stroke_fill)
                # Obtém a caixa delimitadora exata em pixels da renderização do caractere
                x0, y0, x1, y1 = draw.textbbox((cursor_x, cursor_y), char, font=font)
                bboxes.append((x0, y0, x1 - x0, y1 - y0))
                cursor_y += char_h + gap_char
            cursor_x += char_w + gap_col

    else:  # horizontal
        cursor_y = inicio_y
        for linha in todas_colunas:
            cursor_x = inicio_x
            for char in linha:
                draw.text((cursor_x, cursor_y), char, font=font, fill=cor_tinta, stroke_width=2, stroke_fill=stroke_fill)
                # Obtém a caixa delimitadora exata em pixels da renderização do caractere
                x0, y0, x1, y1 = draw.textbbox((cursor_x, cursor_y), char, font=font)
                bboxes.append((x0, y0, x1 - x0, y1 - y0))
                cursor_x += char_w + gap_char
            cursor_y += char_h + gap_col

    bboxes = _expandir_bboxes(bboxes, margem=BBOX_MARGEM)
    return crop, bboxes
    


def bbox_para_yolo(x: int, y: int, w: int, h: int, img_size: int) -> str:
    """
    Converte uma bounding box absoluta (x, y, w, h) para o formato YOLO
    normalizado: <class_id> <cx> <cy> <w> <h>, todos em [0, 1].
    Classe 0 = kanji/hiragana/katakana.
    """
    cx = (x + w / 2) / img_size
    cy = (y + h / 2) / img_size
    wn = w / img_size
    hn = h / img_size
    # Clamp para garantir que valores fiquem dentro de [0, 1]
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    wn = max(0.0, min(1.0, wn))
    hn = max(0.0, min(1.0, hn))
    return f"0 {cx:.6f} {cy:.6f} {wn:.6f} {hn:.6f}"


def save_page(imagem: Image.Image, bboxes: list, img_dir: str, lbl_dir: str, idx: int) -> None:
    """
    Salva a imagem como PNG e o arquivo de label YOLO (.txt).

    Args:
        imagem:  imagem PIL gerada e degradada
        bboxes:  lista de (x, y, w, h) absolutas em pixels
        img_dir: diretório destino das imagens (ex: data/dataset/images/train)
        lbl_dir: diretório destino dos labels  (ex: data/dataset/labels/train)
        idx:     índice único da página (para nomear o arquivo)
    """
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    nome = f"page_{idx:06d}"
    img_path = os.path.join(img_dir, f"{nome}.png")
    lbl_path = os.path.join(lbl_dir, f"{nome}.txt")

    imagem.save(img_path)

    with open(lbl_path, "w", encoding="utf-8") as f:
        for (x, y, w, h) in bboxes:
            f.write(bbox_para_yolo(x, y, w, h, CROP_SIZE) + "\n")


def create_synthetic_manga_images(
    img_dir: str,
    lbl_dir: str,
    amount: int,
    start_idx: int = 0,
) -> int:
    """
    Gera `amount` páginas sintéticas, salvando imagem (.png) e label YOLO (.txt).

    Args:
        img_dir:   diretório destino das imagens
        lbl_dir:   diretório destino dos labels
        amount:    quantidade de páginas a gerar
        start_idx: índice inicial para nomeação dos arquivos

    Returns:
        Número de páginas efetivamente geradas.
    """
    pages_created = 0
    attempts = 0
    max_attempts = amount * 20  # evita loop infinito

    while pages_created < amount and attempts < max_attempts:
        attempts += 1
        crop = get_croped_image()

        resultado = render_kanji(crop)
        if resultado is None:
            continue

        imagem, bboxes = resultado
        if not bboxes:
            continue  # descarta página sem nenhum caractere detectado

        imagem, bboxes = aplicar_degradacoes(imagem, bboxes)
        save_page(imagem, bboxes, img_dir, lbl_dir, start_idx + pages_created)
        pages_created += 1

    return pages_created


if __name__ == "__main__":
    from config import TRAIN_IMG_DIR, TRAIN_LBL_DIR, PAGES_AMOUNT
    print(f"Gerando {PAGES_AMOUNT} páginas de teste em {TRAIN_IMG_DIR}...")
    n = create_synthetic_manga_images(TRAIN_IMG_DIR, TRAIN_LBL_DIR, PAGES_AMOUNT)
    print(f"Geradas: {n} páginas.")