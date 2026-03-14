import os
import shutil
import re
import glob
import numpy as np
import pandas as pd
import laspy
import cv2

from pathlib import Path
from PIL import Image
from ultralytics import YOLO

# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_DIR = BASE_DIR / "INPUT"
OUTPUT_DIR = BASE_DIR / "OUTPUT"
IMAGES_DIR = BASE_DIR / "Images"
CHANNELS_DIR = BASE_DIR / "Channels"
LABELS_DIR = BASE_DIR / "Labels"
COMPLETED_DIR = BASE_DIR / "COMPLETED"
MODEL_PATH = BASE_DIR / "Model" / "best.pt"

CHANNEL0 = CHANNELS_DIR / "channel0"
CHANNEL1 = CHANNELS_DIR / "channel1"
CHANNEL2 = CHANNELS_DIR / "channel2"
CHANNEL3 = CHANNELS_DIR / "channel3"

SLICE_HEIGHT = 0.20
PIXEL_SIZE = 0.01
CONF_THRESHOLD = 0.25

label_codes = {"0":3, "1":1, "2":2, "3":4}

# ============================================================
# UTILITIES
# ============================================================

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def clear_folder(folder: Path, pattern="*"):
    if folder.exists():
        for f in folder.glob(pattern):
            if f.is_file():
                f.unlink()
            else:
                shutil.rmtree(f)

def numeric_suffix(p: Path) -> int:
    return int(re.search(r"_(\d+)$", p.stem).group(1))

def stem_without_suffix(p: Path) -> str:
    return re.sub(r"_(\d+)$", "", p.stem)

def group_by_stem(paths):
    groups = {}
    for p in paths:
        stem = stem_without_suffix(p)
        groups.setdefault(stem, []).append(p)
    return groups

# ============================================================
# 1️⃣ LAS → CHANNEL0 PNG
# ============================================================

def slice_las_to_png(las_path: Path, metadata_store: dict):

    tree_name = las_path.stem
    las = laspy.read(las_path)
    points = np.vstack((las.x, las.y, las.z)).T

    x_min, x_max = np.min(points[:,0]), np.max(points[:,0])
    y_min, y_max = np.min(points[:,1]), np.max(points[:,1])
    z_min, z_max = np.min(points[:,2]), np.max(points[:,2])

    w = int((x_max - x_min) / PIXEL_SIZE) + 1
    h = int((y_max - y_min) / PIXEL_SIZE) + 1
    n_slices = int((z_max - z_min) / SLICE_HEIGHT)

    for i in range(n_slices):
        z_low = z_min + i * SLICE_HEIGHT
        z_high = z_low + SLICE_HEIGHT

        mask = (points[:,2] >= z_low) & (points[:,2] < z_high)
        slice_points = points[mask]

        if len(slice_points) < 5:
            continue

        counts, _, _ = np.histogram2d(
            slice_points[:,0],
            slice_points[:,1],
            bins=[w, h],
            range=[[x_min, x_max],[y_min, y_max]]
        )

        max_val = np.percentile(counts, 99) if np.max(counts) > 0 else 1
        raster = np.clip(counts * (255/(max_val+1e-6)),0,255).astype(np.uint8)

        img_name = f"{tree_name}_slice_{i:03d}.png"
        cv2.imwrite(str(CHANNEL0 / img_name), raster.T)

        metadata_store[img_name] = {
            "tree_name": tree_name,
            "x_origin_global": float(x_min),
            "y_origin_global": float(y_min),
            "z_layer": float(z_low),
            "pixel_size": PIXEL_SIZE,
            "canvas_w": w,
            "canvas_h": h
        }

# ============================================================
# 2️⃣ BUILD CHANNELS
# ============================================================

def build_channel(source: Path, dest: Path):

    ensure_dir(dest)
    clear_folder(dest, "*.png")

    for src in sorted(source.glob("*.png")):
        shutil.copy2(src, dest / src.name)

    groups = group_by_stem(list(dest.glob("*.png")))

    for stem, files in groups.items():

        files_sorted = sorted(files, key=numeric_suffix)

        if files_sorted:
            files_sorted[-1].unlink()
            files_sorted = files_sorted[:-1]

        for file in sorted(files_sorted, key=numeric_suffix, reverse=True):
            new_index = numeric_suffix(file) + 1
            new_name = f"{stem}_{new_index:03d}.png"
            file.rename(dest / new_name)

def build_channel3(source: Path, dest: Path):

    ensure_dir(dest)
    clear_folder(dest, "*.png")

    imgs = sorted(source.glob("*.png"))
    n = len(imgs)

    for i, img_path in enumerate(imgs):
        img = Image.open(img_path)
        w, h = img.size
        gray_val = int(round((i/(n-1))*255)) if n>1 else 0
        arr = np.full((h,w), gray_val, dtype=np.uint8)
        Image.fromarray(arr).save(dest / img_path.name)

# ============================================================
# 3️⃣ BUILD RGBA TIFFS → Images
# ============================================================

def load_gray(path: Path):
    img = Image.open(path)
    return np.array(img.convert("L"), dtype=np.uint8)

def build_multichannel_images():

    ensure_dir(IMAGES_DIR)
    clear_folder(IMAGES_DIR, "*.tif")

    for r_path in CHANNEL0.glob("*.png"):

        stem = r_path.stem
        ch_r = load_gray(r_path)

        g_path = CHANNEL1 / f"{stem}.png"
        b_path = CHANNEL2 / f"{stem}.png"
        a_path = CHANNEL3 / f"{stem}.png"

        ch_g = load_gray(g_path) if g_path.exists() else np.zeros_like(ch_r)
        ch_b = load_gray(b_path) if b_path.exists() else np.zeros_like(ch_r)
        ch_a = load_gray(a_path) if a_path.exists() else np.zeros_like(ch_r)

        stacked = np.stack([ch_r,ch_g,ch_b,ch_a], axis=2)

        Image.fromarray(stacked, mode="RGBA").save(
            IMAGES_DIR / f"{stem}.tif",
            compression="tiff_deflate"
        )

# ============================================================
# 4️⃣ YOLO → Labels
# ============================================================

def run_yolo():

    ensure_dir(LABELS_DIR)
    clear_folder(LABELS_DIR)

    model = YOLO(str(MODEL_PATH))

    model.predict(
        source=str(IMAGES_DIR),
        imgsz=640,
        conf=CONF_THRESHOLD,
        iou=0.7,
        save=False,
        save_txt=True,
        save_conf=True,
        project=str(BASE_DIR),
        name="Labels",
        exist_ok=True,  # ✅ prevents Labels2
        device="cpu"
    )

    # Flatten nested labels folder
    inner = LABELS_DIR / "labels"
    if inner.exists():
        for txt in inner.glob("*.txt"):
            shutil.move(str(txt), str(LABELS_DIR / txt.name))
        shutil.rmtree(inner)

# ============================================================
# 5️⃣ ANNOTATE LAS
# ============================================================

def annotate_las(metadata_store):

    ensure_dir(OUTPUT_DIR)

    for las_file in INPUT_DIR.glob("*.las"):

        tree_name = las_file.stem
        las = laspy.read(las_file)
        label_id = np.zeros(len(las.x), dtype=np.uint8)

        txt_files = list(LABELS_DIR.glob(f"{tree_name}_slice_*.txt"))

        for txt_file in txt_files:

            png_name = txt_file.name.replace(".txt",".png")
            if png_name not in metadata_store:
                continue

            meta = metadata_store[png_name]

            x0 = meta["x_origin_global"]
            y0 = meta["y_origin_global"]
            pix_sz = meta["pixel_size"]
            z_low = meta["z_layer"]
            z_up = z_low + SLICE_HEIGHT
            w = meta["canvas_w"]
            h = meta["canvas_h"]

            labels = pd.read_csv(txt_file, header=None, sep=r"\s+")

            for _, row in labels.iterrows():

                class_id = str(int(row[0]))
                conf = row[5]

                if conf < CONF_THRESHOLD or class_id not in label_codes:
                    continue

                code = label_codes[class_id]

                x_center_pix = row[1]*w
                y_center_pix = row[2]*h
                width_pix = row[3]*w
                height_pix = row[4]*h

                xmin = x0 + (x_center_pix-width_pix/2)*pix_sz
                xmax = x0 + (x_center_pix+width_pix/2)*pix_sz
                ymin = y0 + (y_center_pix-height_pix/2)*pix_sz
                ymax = y0 + (y_center_pix+height_pix/2)*pix_sz

                mask = (
                    (las.z>=z_low)&(las.z<z_up)&
                    (las.x>=xmin)&(las.x<=xmax)&
                    (las.y>=ymin)&(las.y<=ymax)&
                    (label_id==0)
                )

                label_id[mask] = code

        las.classification = label_id
        las.write(OUTPUT_DIR / f"{tree_name}_annotated.las")

# ============================================================
# 6️⃣ MOVE INPUT → COMPLETED
# ============================================================

def move_to_completed(las_file: Path):

    ensure_dir(COMPLETED_DIR)

    base_name = las_file.stem + "_completed"
    target = COMPLETED_DIR / f"{base_name}.las"
    i = 1

    while target.exists():
        target = COMPLETED_DIR / f"{base_name}_{i}.las"
        i += 1

    shutil.move(str(las_file), str(target))

# ============================================================
# 7️⃣ CLEAN WORKING FOLDERS
# ============================================================

def cleanup_working_dirs():

    clear_folder(CHANNEL0)
    clear_folder(CHANNEL1)
    clear_folder(CHANNEL2)
    clear_folder(CHANNEL3)
    clear_folder(IMAGES_DIR)
    clear_folder(LABELS_DIR)

# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    metadata_store = {}

    ensure_dir(CHANNEL0)
    clear_folder(CHANNEL0)

    for las_file in INPUT_DIR.glob("*.las"):
        slice_las_to_png(las_file, metadata_store)

    build_channel(CHANNEL0, CHANNEL1)
    build_channel(CHANNEL1, CHANNEL2)
    build_channel3(CHANNEL0, CHANNEL3)

    build_multichannel_images()
    run_yolo()
    annotate_las(metadata_store)

    for las_file in INPUT_DIR.glob("*.las"):
        move_to_completed(las_file)

    cleanup_working_dirs()

    print("\n✅ FULL PIPELINE COMPLETE — READY FOR NEXT INPUT")

# ============================================================

if __name__ == "__main__":
    main()