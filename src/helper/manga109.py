import os
import random
import glob
from PIL import Image
from config import MANGA109_IMAGES, PAGES_DIR, CROP_SIZE, PAGES_AMOUNT

pages = glob.glob(os.path.join(MANGA109_IMAGES, "**", "*.jpg"), recursive=True)

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

def render_kanji(crop):
    pass


def create_synthetic_manga_images():
    for i in range(PAGES_AMOUNT):
        crop = get_croped_image()



if __name__ == "__main__":
    
    for page in pages:
        print(page)
    create_synthetic_manga_images()