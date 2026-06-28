import os
import glob
import requests
from fontTools.ttLib import TTFont
from src.helper.kanjis import get_kanjis
from config import FONTES_URL, ASSETS_DIR, FONTS_DIR

def download_fonts(fontes, diretorio_destino):
    os.makedirs(diretorio_destino, exist_ok=True)

    for nome, url in fontes.items():
        caminho = os.path.join(diretorio_destino, f"{nome}.ttf")

        if os.path.exists(caminho):
            print(f"{nome} já existe, pulando")
            continue

        try:
            resposta = requests.get(url, timeout=30)
            resposta.raise_for_status()

            with open(caminho, "wb") as f:
                f.write(resposta.content)
            print(f"baixada: {nome}")

        except requests.exceptions.RequestException as e:
            print(f"erro: {nome}: {e}")

def get_fonts_list():
    return glob.glob(os.path.join(FONTS_DIR, "*.ttf"))

def verify_fonts_compatibility(kanji_level = ""):
    kanjis  = get_kanjis(kanji_level)
    font_files = get_fonts_list()

    for font_path in font_files:
        font      = TTFont(font_path)
        font_cmap = font.getBestCmap()

        ausentes = [k for k in kanjis if ord(k) not in font_cmap]

        nome = os.path.basename(font_path)
        if ausentes:
            print(f"{nome}: {len(ausentes)} kanjis ausentes — {''.join(ausentes)}")
        else:
            print(f"{nome}: suporta todos os kanjis {kanji_level}")

if __name__ == "__main__":
    download_fonts(FONTES_URL, FONTS_DIR)
    verify_fonts_compatibility("")