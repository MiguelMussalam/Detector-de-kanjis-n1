import os
import requests

FONTES = {
    "BIZ-UDPGothic-Regular":  "https://raw.githubusercontent.com/google/fonts/main/ofl/bizudpgothic/BIZUDPGothic-Regular.ttf",
    "BIZ-UDPGothic-Bold":     "https://raw.githubusercontent.com/google/fonts/main/ofl/bizudpgothic/BIZUDPGothic-Bold.ttf",
    "BIZ-UDPMincho-Regular":  "https://raw.githubusercontent.com/google/fonts/main/ofl/bizudpmincho/BIZUDPMincho-Regular.ttf",
    "Klee-One-Regular":       "https://raw.githubusercontent.com/google/fonts/main/ofl/kleeone/KleeOne-Regular.ttf",
    "Klee-One-SemiBold":      "https://raw.githubusercontent.com/google/fonts/main/ofl/kleeone/KleeOne-SemiBold.ttf",
    "Hina-Mincho-Regular":    "https://raw.githubusercontent.com/google/fonts/main/ofl/hinamincho/HinaMincho-Regular.ttf",
    "Yusei-Magic-Regular":    "https://raw.githubusercontent.com/google/fonts/main/ofl/yuseimagic/YuseiMagic-Regular.ttf",
    "Dela-Gothic-One":        "https://raw.githubusercontent.com/google/fonts/main/ofl/delagothicone/DelaGothicOne-Regular.ttf",
    "Reggae-One":             "https://raw.githubusercontent.com/google/fonts/main/ofl/reggaeone/ReggaeOne-Regular.ttf",
}

DIRETORIO_SAIDA = os.path.join("assets", "fonts")

def baixar_fontes(fontes, diretorio_destino):
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

if __name__ == "__main__":
    baixar_fontes(FONTES, DIRETORIO_SAIDA)