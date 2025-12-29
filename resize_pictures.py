import os
from PIL import Image

folder = "datasets/coco_bumblebee/images/test"

for filename in os.listdir(folder):
    if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
        path = os.path.join(folder, filename)
        img = Image.open(path)
        img = img.resize((224, 224), Image.LANCZOS)
        img.save(path)
        print(f"{filename} skaliert auf 224x224")