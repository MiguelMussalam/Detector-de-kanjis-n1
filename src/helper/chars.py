from fontTools.ttLib import TTFont
from src.helper.kanjis import get_kanjis
from src.helper.fonts import get_fonts_list
import glob


def is_japanese(codepoint):
    return (
        0x3041 <= codepoint <= 0x3096 or  # hiragana
        0x30A1 <= codepoint <= 0x30F6 or  # katakana
        0x4E00 <= codepoint <= 0x9FFF     # kanji
    )

def get_supported_kanjis_from_fonts():
    codepoints_list = []
    font_list = get_fonts_list()

    if not font_list:
        raise FileNotFoundError(
            "Nenhuma fonte (.ttf) encontrada na pasta de assets/fonts/. "
            "Verifique se o download das fontes foi executado corretamente."
        )

    for font in font_list:
        cmap = TTFont(font).getBestCmap()
        codepoints_list.append(set(cmap.keys()))

    chars_comuns = set.intersection(*codepoints_list)

    return [chr(c) for c in chars_comuns if is_japanese(c)]


if __name__ == "__main__":
    kanjis_suportados = get_supported_kanjis_from_fonts()

    print(kanjis_suportados)
    print(f"Total de {len(kanjis_suportados)} kanjis suportados entre todas as fontes.")