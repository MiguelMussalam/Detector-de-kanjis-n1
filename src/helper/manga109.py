import os
import random
import glob
from PIL import Image, ImageDraw, ImageFont
from config import MANGA109_IMAGES, PAGES_DIR, CROP_SIZE, PAGES_AMOUNT, FONTS_DIR
from src.helper.kanjis import get_kanjis

pages = glob.glob(os.path.join(MANGA109_IMAGES, "**", "*.jpg"), recursive=True)
fonts = glob.glob(os.path.join(FONTS_DIR, "*.ttf"))
kanjis = list(get_kanjis(''))

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

def render_kanji(crop):
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

    inicio_x = random.randint(0, max_x)
    inicio_y = random.randint(0, max_y)

    # gera caracteres para todas as colunas
    todas_colunas = []
    for _ in range(n_colunas):
        coluna = [random.choice(kanjis) for _ in range(chars_por_coluna)]
        todas_colunas.append(coluna)

    # calcula tamanho de um caractere
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
                draw.text((cursor_x, cursor_y), char, font=font, fill=(0, 0, 0))
                bboxes.append((cursor_x, cursor_y, char_w, char_h))
                cursor_y += char_h + gap_char
            cursor_x += char_w + gap_col

    else:  # horizontal
        cursor_y = inicio_y
        for linha in todas_colunas:
            cursor_x = inicio_x
            for char in linha:
                draw.text((cursor_x, cursor_y), char, font=font, fill=(0, 0, 0))
                bboxes.append((cursor_x, cursor_y, char_w, char_h))
                cursor_x += char_w + gap_char
            cursor_y += char_h + gap_col

    return crop, bboxes
    


def create_synthetic_manga_images():
    pages_created = 0
    while pages_created < PAGES_AMOUNT:
        crop = get_croped_image()

        resultado = render_kanji(crop)
        if resultado is None:
            continue
        else:
            pages_created += 1

        imagem, bboxes = resultado
        imagem.show()



if __name__ == "__main__":
    
    for page in pages:
        print(page)
    create_synthetic_manga_images()