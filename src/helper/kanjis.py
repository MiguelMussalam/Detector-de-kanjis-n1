import os
import json
import requests
from config import KANJI_DATA_URL, KANJI_DATA_CACHE

kanji_levels: dict = {
    "n1": 1,
    "n2": 2,
    "n3": 3,
    "n4": 4,
    "n5": 5
}


def get_kanjis(kanji_level) -> list:
    if os.path.exists(KANJI_DATA_CACHE):
        with open(KANJI_DATA_CACHE, "r", encoding="utf-8") as f:
            kanjis = json.load(f)
    else:
        print(f"Baixando dados de kanjis de {KANJI_DATA_URL}...")
        response = requests.get(KANJI_DATA_URL)
        response.raise_for_status()
        kanjis = response.json()
        
        os.makedirs(os.path.dirname(KANJI_DATA_CACHE), exist_ok=True)
        with open(KANJI_DATA_CACHE, "w", encoding="utf-8") as f:
            json.dump(kanjis, f, ensure_ascii=False, indent=4)
        print(f"Dados de kanjis salvos em cache: {KANJI_DATA_CACHE}")

    if (kanji_level == ''):
        return list(kanjis.keys())
    else:
        return list({kanji: data for kanji, data in kanjis.items() if data.get("jlpt_new") == kanji_levels.get(kanji_level)}.keys())


if __name__ == "__main__":
    kanji_level = ""
    kanjis = get_kanjis(kanji_level)
    print('total kanjis ', kanji_level + ':', len(kanjis))
    print('Kanjis', kanji_level + ':')
    print(kanjis[:10], "... (truncado)")