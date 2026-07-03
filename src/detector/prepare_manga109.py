"""
prepare_manga109.py
===================
Converte o dataset Manga109 bruto para o formato YOLOv8
com bboxes em nível de bloco de texto.

Uso:
    python -m src.detector.prepare_manga109
"""

import os
import glob
import zipfile
import xml.etree.ElementTree as ET
from PIL import Image
import yaml

from config import (
    ROOT_DIR,
    MANGA109_DIR,
    MANGA109_IMAGES,
    MANGA109_ANNOTATIONS,
    MANGA109_YOLO_DIR,
    MANGA109_VAL_VOLUMES
)

def extract_annotations_if_needed():
    """Extrai os arquivos XML do zip caso a pasta de anotações não exista."""
    if not os.path.exists(MANGA109_ANNOTATIONS) or len(os.listdir(MANGA109_ANNOTATIONS)) == 0:
        print(f"[INFO] Diretório de anotações não encontrado ou vazio: {MANGA109_ANNOTATIONS}")
        
        # Procurar o zip no diretório raiz do projeto
        zip_path = os.path.join(ROOT_DIR, "Manga109_released_2023_12_07.zip")
        if not os.path.exists(zip_path):
            raise FileNotFoundError(
                f"Arquivo zip do Manga109 não encontrado em: {zip_path}\n"
                "Por favor, coloque o arquivo zip na raiz do projeto."
            )
        
        print(f"[INFO] Extraindo anotações do zip: {zip_path}...")
        os.makedirs(MANGA109_ANNOTATIONS, exist_ok=True)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            extracted_count = 0
            for name in zip_ref.namelist():
                # Filtrar arquivos XML da pasta de anotações principal do release
                if "annotations/" in name and name.endswith(".xml") and not "__MACOSX" in name:
                    filename = os.path.basename(name)
                    if filename:  # Evita diretórios vazios
                        dest_path = os.path.join(MANGA109_ANNOTATIONS, filename)
                        with open(dest_path, "wb") as f_out:
                            f_out.write(zip_ref.read(name))
                        extracted_count += 1
            print(f"[INFO] {extracted_count} arquivos XML extraídos com sucesso.")
    else:
        print(f"[INFO] Anotações já existentes em {MANGA109_ANNOTATIONS}")

def preparar_manga109():
    extract_annotations_if_needed()
    
    print("[INFO] Iniciando preparação do dataset Manga109...")
    
    # Criar estrutura de saída do YOLO
    for split in ["train", "val"]:
        os.makedirs(os.path.join(MANGA109_YOLO_DIR, "images", split), exist_ok=True)
        os.makedirs(os.path.join(MANGA109_YOLO_DIR, "labels", split), exist_ok=True)
        
    xml_files = glob.glob(os.path.join(MANGA109_ANNOTATIONS, "*.xml"))
    if not xml_files:
        raise FileNotFoundError(f"Nenhum arquivo XML encontrado em {MANGA109_ANNOTATIONS}")
        
    print(f"[INFO] Total de volumes a processar: {len(xml_files)}")
    
    total_imagens_salvas = 0
    
    for idx_vol, xml_path in enumerate(xml_files, 1):
        volume_name = os.path.splitext(os.path.basename(xml_path))[0]
        split = "val" if volume_name in MANGA109_VAL_VOLUMES else "train"
        
        print(f"[{idx_vol}/{len(xml_files)}] Processando volume '{volume_name}' ({split})...")
        
        # Carregar anotações XML
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
        except Exception as e:
            print(f"[ERROR] Falha ao ler XML {xml_path}: {e}")
            continue
            
        # Percorrer páginas
        for page in root.findall(".//page"):
            page_index_str = page.attrib.get("index")
            if page_index_str is None:
                continue
                
            page_index = int(page_index_str)
            page_width = float(page.attrib.get("width", 1.0))
            page_height = float(page.attrib.get("height", 1.0))
            
            # Caminho da imagem correspondente
            img_src_path = os.path.join(MANGA109_IMAGES, volume_name, f"{page_index:03d}.jpg")
            if not os.path.exists(img_src_path):
                # Alguns volumes podem ter convenções ligeiramente diferentes ou faltar páginas
                continue
                
            # Extrair bboxes de texto (<text>)
            bboxes = []
            for text_elem in page.findall("text"):
                xmin = float(text_elem.attrib.get("xmin", 0))
                ymin = float(text_elem.attrib.get("ymin", 0))
                xmax = float(text_elem.attrib.get("xmax", 0))
                ymax = float(text_elem.attrib.get("ymax", 0))
                bboxes.append((xmin, ymin, xmax, ymax))
                
            # Salvar imagem redimensionada
            img_dest_name = f"{volume_name}_page_{page_index:03d}.jpg"
            img_dest_path = os.path.join(MANGA109_YOLO_DIR, "images", split, img_dest_name)
            
            try:
                with Image.open(img_src_path) as img:
                    img_rgb = img.convert("RGB")
                    img_resized = img_rgb.resize((640, 640), Image.Resampling.LANCZOS)
                    img_resized.save(img_dest_path, "JPEG", quality=95)
            except Exception as e:
                print(f"[ERROR] Falha ao processar imagem {img_src_path}: {e}")
                continue
                
            # Salvar label YOLO
            lbl_dest_name = f"{volume_name}_page_{page_index:03d}.txt"
            lbl_dest_path = os.path.join(MANGA109_YOLO_DIR, "labels", split, lbl_dest_name)
            
            scale_x = 640.0 / page_width
            scale_y = 640.0 / page_height
            
            with open(lbl_dest_path, "w", encoding="utf-8") as f_out:
                for xmin, ymin, xmax, ymax in bboxes:
                    # Redimensionar bboxes
                    xmin_new = xmin * scale_x
                    ymin_new = ymin * scale_y
                    xmax_new = xmax * scale_x
                    ymax_new = ymax * scale_y
                    
                    # YOLO normalizado
                    cx = (xmin_new + xmax_new) / 2.0 / 640.0
                    cy = (ymin_new + ymax_new) / 2.0 / 640.0
                    w  = (xmax_new - xmin_new) / 640.0
                    h  = (ymax_new - ymin_new) / 640.0
                    
                    # Clamp para manter dentro dos limites de [0, 1]
                    cx = max(0.0, min(1.0, cx))
                    cy = max(0.0, min(1.0, cy))
                    w  = max(0.0, min(1.0, w))
                    h  = max(0.0, min(1.0, h))
                    
                    f_out.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
                    
            total_imagens_salvas += 1
            
    # Gerar data.yaml
    yaml_path = os.path.join(MANGA109_YOLO_DIR, "data.yaml")
    config = {
        "path": os.path.abspath(MANGA109_YOLO_DIR),
        "train": "images/train",
        "val":   "images/val",
        "nc":    1,
        "names": ["text"]
    }
    
    with open(yaml_path, "w", encoding="utf-8") as f_yaml:
        yaml.dump(config, f_yaml, allow_unicode=True, default_flow_style=False)
        
    print("-" * 60)
    print("Preparação concluída!")
    print(f"Total de imagens salvas: {total_imagens_salvas}")
    print(f"data.yaml gerado em: {yaml_path}")
    print(f"Diretório de saída: {MANGA109_YOLO_DIR}")

if __name__ == "__main__":
    preparar_manga109()
