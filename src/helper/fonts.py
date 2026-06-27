import os
import requests
from fontTools.ttLib import TTFont
from kanjis import get_kanjis
from config import FONTES_URL, ASSETS_DIR

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

def verify_fonts_compatibility():
    kanjis_n1 = get_kanjis("n1")
    for fonts in os.listdir(ASSETS_DIR):
        kanjis_not_supported = []
        font = TTFont(os.path.join(ASSETS_DIR, fonts))
        font_cmap = font.getBestCmap()
        for kanji in kanjis_n1:
            if ord(kanji) not in font_cmap:
                kanjis_not_supported.append(kanji)
        if kanjis_not_supported:
            print(f"Fonte {fonts} não suporta os seguintes kanjis: {', '.join(kanjis_not_supported)}")
        else:
            print(f"Fonte {fonts} suporta todos os kanjis N1")


if __name__ == "__main__":
    download_fonts(FONTES_URL, ASSETS_DIR)
    verify_fonts_compatibility()