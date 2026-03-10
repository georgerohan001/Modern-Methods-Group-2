"""Train YOLO11 with 4-channel multi-channel data."""

import torch
from ultralytics import YOLO

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA device:", torch.cuda.get_device_name(0))

model = YOLO("yolo11s_4ch.pt")
print(f"Model input channels: {model.model.model[0].conv.in_channels}")

model.train(
    data="data/datasets/datasets/my_yolo_dataset/data_multichannel.yaml",
    epochs=10,
    batch=32,
    imgsz=640,
    name="tree_multichannel_test",
    project="data/runs",
    exist_ok=True,
    patience=5,
    verbose=True,
    device=0 if torch.cuda.is_available() else "cpu",
)