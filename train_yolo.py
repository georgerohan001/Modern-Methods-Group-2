import os
import numpy as np
import torch
from ultralytics import YOLO

# -----------------------------
# Print CUDA information
# -----------------------------
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA device name:", torch.cuda.get_device_name(0))
else:
    print("No CUDA device found.")

# -----------------------------
# Change working directory
# -----------------------------
dataset_path = r"C:\Users\HiWi\Desktop\Group 1 Project Folder\datasets"
os.chdir(dataset_path)
print("Current working directory:", os.getcwd())

# -----------------------------
# YOLO Training
# -----------------------------
model = YOLO("yolo11s.pt")

model.train(
    data="data.yaml",
    epochs=300,
    batch=32,
    imgsz=640,
    name="tree_cross_section_5ch_gpu",
    cache=True,
    patience=30,
    lr0=0.01,
    rect=True,
    augment=True,
)