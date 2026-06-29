import os
import random
import glob
from PIL import Image, ImageDraw, ImageFont, ImageStat
from config import MANGA109_IMAGES, PAGES_DIR, CROP_SIZE, PAGES_AMOUNT, FONTS_DIR, LIMITE_DESVIO_REGIAO, MAX_TENTATIVAS_POSICAO
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
        imagem = aplicar_degradacoes(imagem)
        imagem.show()



if __name__ == "__main__":
    
    for page in pages:
        print(page)
    create_synthetic_manga_images()