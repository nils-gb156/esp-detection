from ultralytics import YOLO

# Modell laden
model = YOLO("examples/bumblebee_detection/espdet_pico_224_224_bumblebee.pt")

# Inferenz auf allen Testbildern im Ordner
results = model.predict(
    source="datasets/coco_bumblebee/images/test",  # Pfad zum Test-Ordner
    conf=0.4,   # Confidence Threshold, z.B. 0.3
    save=True,  # speichert Bilder mit Bounding Boxes
    show=False  # kein Fenster öffnen
)

# Ergebnisse ausgeben
print(f"\nInsgesamt {len(results)} Bilder getestet:\n")
for i, r in enumerate(results):
    print(f"Bild {i+1}: {len(r.boxes)} Hummeln gefunden")
    if len(r.boxes) > 0:
        print(f"  Konfidenz: {r.boxes.conf.tolist()}")
        
print(f"\nGespeichert in: runs/detect/predict/")
