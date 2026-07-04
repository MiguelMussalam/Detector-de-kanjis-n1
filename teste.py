from ultralytics import YOLO


model = YOLO("yolo26n.pt")
resultado = model.predict(source="data/raw/manga109/images/AisazuNihaIrarenai/001.jpg", save= True)
