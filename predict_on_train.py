from ultralytics import YOLO
from pathlib import Path

MODEL_PATH = Path(
    r"C:\Users\HiWi\Desktop\Group 1 Project Folder\datasets\runs\detect\train14\weights\best.pt"
)

SOURCE_DIR = Path(
    r"C:\Users\HiWi\Desktop\Group 1 Project Folder\datasets\my_yolo_dataset\images\train"
)

model = YOLO(str(MODEL_PATH))

model.predict(
    source=str(SOURCE_DIR),
    imgsz=640,
    conf=0.25,
    iou=0.7,
    save=True,
    save_txt=True,     # ✅ YOLO format labels
    save_conf=True,    # ✅ include confidence scores
    device=0,
)

print("Prediction complete.")