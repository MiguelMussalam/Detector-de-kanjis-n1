import os
import requests
from fontTools.ttLib import TTFont
from kanjis import get_kanjis

FONTES = {
    "Shippori Antique":         "https://raw.githubusercontent.com/fontdasu/ShipporiAntique/master/fonts/ttf/ShipporiAntique-Regular.ttf",
    "BIZ-UDPGothic-Regular":    "https://raw.githubusercontent.com/google/fonts/main/ofl/bizudpgothic/BIZUDPGothic-Regular.ttf",
    "BIZ-UDPGothic-Bold":       "https://raw.githubusercontent.com/google/fonts/main/ofl/bizudpgothic/BIZUDPGothic-Bold.ttf",
    "BIZ-UDPMincho-Regular":    "https://raw.githubusercontent.com/google/fonts/main/ofl/bizudpmincho/BIZUDPMincho-Regular.ttf",
    "Klee-One-Regular":         "https://raw.githubusercontent.com/google/fonts/main/ofl/kleeone/KleeOne-Regular.ttf",
    "Klee-One-SemiBold":        "https://raw.githubusercontent.com/google/fonts/main/ofl/kleeone/KleeOne-SemiBold.ttf",
    "Hina-Mincho-Regular":      "https://raw.githubusercontent.com/google/fonts/main/ofl/hinamincho/HinaMincho-Regular.ttf",
    "Yusei-Magic-Regular":      "https://raw.githubusercontent.com/google/fonts/main/ofl/yuseimagic/YuseiMagic-Regular.ttf",
    "Dela-Gothic-One":          "https://raw.githubusercontent.com/google/fonts/main/ofl/delagothicone/DelaGothicOne-Regular.ttf",
    "Reggae-One":               "https://raw.githubusercontent.com/google/fonts/main/ofl/reggaeone/ReggaeOne-Regular.ttf",
}

DIRETORIO_SAIDA = os.path.join("assets", "fonts")

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
    for fonts in os.listdir(DIRETORIO_SAIDA):
        kanjis_not_supported = []
        font = TTFont(os.path.join(DIRETORIO_SAIDA, fonts))
        font_cmap = font.getBestCmap()
        for kanji in kanjis_n1:
            if ord(kanji) not in font_cmap:
                kanjis_not_supported.append(kanji)
        if kanjis_not_supported:
            print(f"Fonte {fonts} não suporta os seguintes kanjis: {', '.join(kanjis_not_supported)}")
        else:
            print(f"Fonte {fonts} suporta todos os kanjis N1")



if __name__ == "__main__":
    download_fonts(FONTES, DIRETORIO_SAIDA)
    verify_fonts_compatibility()