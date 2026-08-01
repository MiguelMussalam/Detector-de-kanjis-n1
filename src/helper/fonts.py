import io
import os
import glob
import zipfile
import requests
from fontTools.ttLib import TTFont
from src.helper.kanjis import get_kanjis
from config import FONTES_URL, ASSETS_DIR, FONTS_DIR

def download_fonts(fontes, diretorio_destino):
    """
    `fontes[nome]` pode ser uma URL direta do arquivo (.ttf/.otf, salvo como
    está) ou uma tupla `(url_do_zip, caminho_dentro_do_zip)` -- algumas
    fontes japonesas gratuitas (ex: a família 源暎 de okoneya.jp) só são
    distribuídas em .zip, não como arquivo solto.
    """
    os.makedirs(diretorio_destino, exist_ok=True)

    for nome, origem in fontes.items():
        caminho = os.path.join(diretorio_destino, f"{nome}.ttf")

        if os.path.exists(caminho):
            print(f"{nome} já existe, pulando")
            continue

        url, caminho_no_zip = origem if isinstance(origem, tuple) else (origem, None)

        try:
            resposta = requests.get(url, timeout=30)
            resposta.raise_for_status()

            if caminho_no_zip:
                with zipfile.ZipFile(io.BytesIO(resposta.content)) as z:
                    conteudo = z.read(caminho_no_zip)
            else:
                conteudo = resposta.content

            with open(caminho, "wb") as f:
                f.write(conteudo)
            print(f"baixada: {nome}")

        except (requests.exceptions.RequestException, KeyError, zipfile.BadZipFile) as e:
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