import random
import glob
from PIL import Image
from config import MANGA109_IMAGES

pages = glob.glob("data/raw/Manga109/images/**/*.jpg", recursive=True)


def generate_cropped_images():
    pass

def create_synthetic_manga_images():
    pass


if __name__ == "__main__":
    print(pages)