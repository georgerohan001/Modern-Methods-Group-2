from ultralytics import YOLO
from pathlib import Path

CUSTOM_YAML = Path(r"C:\Users\HiWi\Desktop\Group 1 Project Folder\datasets\yolo11.yaml")
DATA_YAML   = Path(r"C:\Users\HiWi\Desktop\Group 1 Project Folder\datasets\data.yaml")

def main():
    model = YOLO(str(CUSTOM_YAML))

    # Verify 4-channel model
    print("First conv shape:",
          model.model.model[0].conv.weight.shape)

    model.train(
        data=str(DATA_YAML),
        epochs=300,
        batch=32,
        imgsz=640,
        device=0,      # GPU
        amp=True,
    )

if __name__ == "__main__":
    main()