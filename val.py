from ultralytics import YOLO
from nn.esp_tasks import custom_parse_model
import ultralytics.nn.tasks as tasks

tasks.parse_model = custom_parse_model # add ESP-customized block
# load a model
model = YOLO('examples/bumblebee_detection/espdet_pico_224_224_bumblebee.pt') # load an esp-detection model
# model = YOLO("path/to/best.pt") # load a custom model

# validate the model
metrics = model.val(data='cfg/datasets/coco_bumblebee.yaml', save_json=True)
# metrics = model.val(data='cfg/datasets/coco_bumblebee.yaml', rect=True, imgsz=[160, 288])
metrics.box.map # map50-95
metrics.box.map50 # map50
metrics.box.map75 # map75
metrics.box.maps # a list contains map50-95 of each category
# predict
# Perform object detection on an image
results = model("examples/bumblebee_detection/bumblebee.jpg")
results[0].show()