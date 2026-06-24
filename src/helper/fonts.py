import requests

kanji_levels: dict = {
    "n1": 1,
    "n2": 2,
    "n3": 3,
    "n4": 4,
    "n5": 5
}


def get_kanjis(kanji_level) -> dict:
    url = "https://raw.githubusercontent.com/davidluzgouveia/kanji-data/master/kanji.json"
    response = requests.get(url)
    kanjis = response.json()
    return {kanji: data for kanji, data in kanjis.items() if data.get("jlpt_new") == kanji_levels.get(kanji_level)}.keys()


if __name__ == "__main__":
    kanjis = get_kanjis("n1")
    print(kanjis)