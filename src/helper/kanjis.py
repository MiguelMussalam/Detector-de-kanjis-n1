import requests

URL = "https://raw.githubusercontent.com/davidluzgouveia/kanji-data/master/kanji.json"
kanji_levels: dict = {
    "n1": 1,
    "n2": 2,
    "n3": 3,
    "n4": 4,
    "n5": 5
}


def get_kanjis(kanji_level) -> dict:
    response = requests.get(URL)
    kanjis = response.json()

    if (kanji_level == ''):
        return kanjis.keys()
    else:
        return {kanji: data for kanji, data in kanjis.items() if data.get("jlpt_new") == kanji_levels.get(kanji_level)}.keys()


if __name__ == "__main__":
    kanji_level = ""
    kanjis = get_kanjis(kanji_level)
    print('total kanjis ', kanji_level + ':', len(kanjis))
    print('Kanjis', kanji_level + ':')
    print(kanjis)