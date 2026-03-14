from __future__ import annotations

import argparse
import csv
import itertools
import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import laspy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent

INPUT_DIR = BASE_DIR / "INPUT"
OUTPUT_DIR = BASE_DIR / "OUTPUT"
IMAGES_DIR = BASE_DIR / "Images"
CHANNELS_DIR = BASE_DIR / "Channels"
TEST_DIR = BASE_DIR / "TEST"
PREDICTED_DIR = BASE_DIR / "PREDICTED"
MODEL_PATH = BASE_DIR / "Model" / "best.pt"

CHANNEL0 = CHANNELS_DIR / "channel0"
CHANNEL1 = CHANNELS_DIR / "channel1"
CHANNEL2 = CHANNELS_DIR / "channel2"
CHANNEL3 = CHANNELS_DIR / "channel3"

SLICE_HEIGHT = 0.20
PIXEL_SIZE = 0.01
CONF_THRESHOLD = 0.10


@dataclass
class ClassConfig:
    name: str
    code: int
    color: tuple[int, int, int]
    priority: int


CLASSES = {
    0: ClassConfig("twigs", 3, (120, 120, 120), 2),
    1: ClassConfig("trunk", 1, (0, 255, 0), 4),
    2: ClassConfig("branch", 2, (255, 0, 0), 3),
    3: ClassConfig("grass", 4, (0, 255, 255), 1),
}

NUM_CLASSES = len(CLASSES)
BACKGROUND_IDX = NUM_CLASSES
YOLO_CLASS_NAMES = [cfg.name for _, cfg in sorted(CLASSES.items())]
CLASS_COLORS = {i: cfg.color for i, cfg in CLASSES.items()}
CLASS_PRIORITIES = {i: cfg.priority for i, cfg in CLASSES.items()}
ASSIGNED_PIXELS_DIR = BASE_DIR / "Assigned"

parser = argparse.ArgumentParser()
parser.add_argument("--tree", action="append", default=None)
parser.add_argument("--conf", type=float, default=CONF_THRESHOLD)
parser.add_argument("--output-root", type=Path, default=OUTPUT_DIR / "pixel_metrics")
parser.add_argument("--assigned-pixels-dir", type=Path, default=ASSIGNED_PIXELS_DIR)
args = parser.parse_args()

include_trees = set(args.tree or [])

CHANNEL0.mkdir(parents=True, exist_ok=True)
CHANNEL1.mkdir(parents=True, exist_ok=True)
CHANNEL2.mkdir(parents=True, exist_ok=True)
CHANNEL3.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
PREDICTED_DIR.mkdir(parents=True, exist_ok=True)
args.output_root.mkdir(parents=True, exist_ok=True)
(args.output_root / "confusion_matrices").mkdir(parents=True, exist_ok=True)
args.assigned_pixels_dir.mkdir(parents=True, exist_ok=True)

metadata_store = {}

tree_folders = sorted(
    d
    for d in TEST_DIR.iterdir()
    if d.is_dir()
    and (d / "obj.names").exists()
    and ((not include_trees) or d.name in include_trees)
)
print(
    f"\nFound {len(tree_folders)} tree(s) with reference labels: {[f.name for f in tree_folders]}"
)

las_trees = {path.stem for path in INPUT_DIR.glob("*.las")}
print(f"Found LAS file(s) in INPUT: {las_trees}")

print("\n[1/3] Preparing source images...")

# If source images already exist, skip regeneration
las_files = sorted(INPUT_DIR.glob("*.las"))
expected_images = 0
for las_file in las_files:
    las_tmp = laspy.read(las_file)
    points_tmp = np.column_stack((las_tmp.x, las_tmp.y, las_tmp.z))
    if points_tmp.size == 0:
        continue
    z_min_tmp = np.min(points_tmp[:, 2])
    z_max_tmp = np.max(points_tmp[:, 2])
    expected_images += int((z_max_tmp - z_min_tmp) / SLICE_HEIGHT)

existing_images = len(list(IMAGES_DIR.glob("*.tif")))
if expected_images > 0 and existing_images >= expected_images:
    print(f"   Reusing {existing_images} existing images in {IMAGES_DIR}")
else:
    for las_file in las_files:
        tree_name = las_file.stem
        las = laspy.read(las_file)
        points = np.column_stack((las.x, las.y, las.z))

        x_min, y_min, z_min = np.min(points, axis=0)
        x_max, y_max, z_max = np.max(points, axis=0)

        w = int((x_max - x_min) / PIXEL_SIZE) + 1
        h = int((y_max - y_min) / PIXEL_SIZE) + 1
        n_slices = int((z_max - z_min) / SLICE_HEIGHT)

        for i in range(n_slices):
            z_low = z_min + i * SLICE_HEIGHT
            z_high = z_low + SLICE_HEIGHT

            mask = (points[:, 2] >= z_low) & (points[:, 2] < z_high)
            slice_points = points[mask]

            counts, _, _ = np.histogram2d(
                slice_points[:, 0],
                slice_points[:, 1],
                bins=[w, h],
                range=[[x_min, x_max], [y_min, y_max]],
            )

            max_val = np.percentile(counts, 99)
            raster = np.clip(counts * (255 / (max_val + 1e-6)), 0, 255).astype(np.uint8)

            img_name = f"{tree_name}_slice_{i:03d}.png"
            cv2.imwrite(str(CHANNEL0 / img_name), raster.T)

            metadata_store[img_name] = {
                "tree_name": tree_name,
                "x_origin_global": float(x_min),
                "y_origin_global": float(y_min),
                "z_layer": float(z_low),
                "pixel_size": PIXEL_SIZE,
                "canvas_w": w,
                "canvas_h": h,
            }

if expected_images > 0 and existing_images >= expected_images:
    pass
else:
    channel0_files = sorted(
        CHANNEL0.glob("*.png"),
        key=lambda p: (
            re.sub(r"_(\d+)$", "", p.stem),
            int(re.search(r"_(\d+)$", p.stem).group(1))
            if re.search(r"_(\d+)$", p.stem)
            else 0,
        ),
    )

    grouped = {}
    for path in channel0_files:
        stem = re.sub(r"_(\d+)$", "", path.stem)
        grouped.setdefault(stem, []).append(path)

    for stem, paths in grouped.items():
        ordered_paths = sorted(
            paths,
            key=lambda p: (
                int(re.search(r"_(\d+)$", p.stem).group(1))
                if re.search(r"_(\d+)$", p.stem)
                else 0
            ),
        )
        for i, dst in enumerate(ordered_paths):
            src = ordered_paths[max(i - 1, 0)]
            shutil.copy2(src, CHANNEL1 / dst.name)

    channel1_files = sorted(
        CHANNEL1.glob("*.png"),
        key=lambda p: (
            re.sub(r"_(\d+)$", "", p.stem),
            int(re.search(r"_(\d+)$", p.stem).group(1))
            if re.search(r"_(\d+)$", p.stem)
            else 0,
        ),
    )

    grouped = {}
    for path in channel1_files:
        stem = re.sub(r"_(\d+)$", "", path.stem)
        grouped.setdefault(stem, []).append(path)

    for stem, paths in grouped.items():
        ordered_paths = sorted(
            paths,
            key=lambda p: (
                int(re.search(r"_(\d+)$", p.stem).group(1))
                if re.search(r"_(\d+)$", p.stem)
                else 0
            ),
        )
        for i, dst in enumerate(ordered_paths):
            src = ordered_paths[max(i - 1, 0)]
            shutil.copy2(src, CHANNEL2 / dst.name)

    imgs = sorted(CHANNEL0.glob("*.png"))
    total = len(imgs)
    for i, img_path in enumerate(imgs):
        img = Image.open(img_path)
        w, h = img.size
        gray_val = 0 if total <= 1 else int(round((i / (total - 1)) * 255))
        arr = np.full((h, w), gray_val, dtype=np.uint8)
        Image.fromarray(arr).save(CHANNEL3 / img_path.name)

    for r_path in CHANNEL0.glob("*.png"):
        stem = r_path.stem
        ch_r = np.array(Image.open(r_path).convert("L"), dtype=np.uint8)
        ch_g = np.array(
            Image.open(CHANNEL1 / f"{stem}.png").convert("L"), dtype=np.uint8
        )
        ch_b = np.array(
            Image.open(CHANNEL2 / f"{stem}.png").convert("L"), dtype=np.uint8
        )
        ch_a = np.array(
            Image.open(CHANNEL3 / f"{stem}.png").convert("L"), dtype=np.uint8
        )
        stacked = np.stack([ch_r, ch_g, ch_b, ch_a], axis=2)
        Image.fromarray(stacked).save(
            IMAGES_DIR / f"{stem}.tif", compression="tiff_deflate"
        )

    print(f"   Generated {len(list(IMAGES_DIR.glob('*.tif')))} images")

print("\n[2/3] Preparing predicted labels...")

# Reuse predictions or generate new
images = sorted(IMAGES_DIR.glob("*.tif"))
image_stems = {p.stem for p in images}
pred_direct = {p.stem for p in PREDICTED_DIR.glob("*.txt")}

if image_stems and image_stems.issubset(pred_direct):
    print("Reusing existing labels")
else:
    model = YOLO(str(MODEL_PATH))
    model.predict(
        source=str(IMAGES_DIR),
        imgsz=640,
        conf=args.conf,
        iou=0.7,
        save=False,
        save_txt=True,
        save_conf=True,
        project=str(PREDICTED_DIR.parent),
        name=PREDICTED_DIR.name,
        exist_ok=True,
        device="cpu",
    )

for stem in image_stems:
    (PREDICTED_DIR / f"{stem}.txt").touch()

print(f"   Using {sum(1 for _ in PREDICTED_DIR.glob('*.txt'))} prediction file(s)")

print("\n[3/3] Calculating metrics...")

tree_summaries = []
total_confusion_counts = np.zeros((NUM_CLASSES + 1, NUM_CLASSES + 1), dtype=np.int64)

for tree_folder in tree_folders:
    tree_name = tree_folder.name
    print(f"\n>>> Tree: {tree_name}")

    gt_dir = tree_folder / "obj_train_data"
    gt_files = sorted(gt_dir.glob("*.txt"))
    counts = np.zeros((NUM_CLASSES + 1, NUM_CLASSES + 1), dtype=np.int64)

    for gt_file in gt_files:
        stem = gt_file.stem
        pred_stem = stem[5:] if stem.startswith("tree_") else stem
        # Prefer direct .txt in PREDICTED_DIR, fall back to PREDICTED_DIR/labels
        pred_file = PREDICTED_DIR / f"{pred_stem}.txt"
        if not pred_file.exists():
            alt = PREDICTED_DIR / "labels" / f"{pred_stem}.txt"
            if alt.exists():
                pred_file = alt

        with Image.open(CHANNEL0 / f"{pred_stem}.png") as image_obj:
            width, height = image_obj.size

        if (not gt_file.exists()) or gt_file.stat().st_size == 0:
            gt_arr = np.empty((0, 5), dtype=float)
        else:
            gt_arr = np.loadtxt(gt_file, dtype=float)
            if gt_arr.size == 0:
                gt_arr = np.empty((0, 5), dtype=float)
            else:
                gt_arr = np.atleast_2d(gt_arr)
                if gt_arr.shape[1] < 5:
                    padded = np.zeros((gt_arr.shape[0], 5), dtype=float)
                    padded[:, : gt_arr.shape[1]] = gt_arr
                    gt_arr = padded

        if (not pred_file.exists()) or pred_file.stat().st_size == 0:
            pred_arr = np.empty((0, 6), dtype=float)
        else:
            pred_arr = np.loadtxt(pred_file, dtype=float)
            if pred_arr.size == 0:
                pred_arr = np.empty((0, 6), dtype=float)
            else:
                pred_arr = np.atleast_2d(pred_arr)
                if pred_arr.shape[1] < 6:
                    padded = np.zeros((pred_arr.shape[0], 6), dtype=float)
                    padded[:, : pred_arr.shape[1]] = pred_arr
                    pred_arr = padded

        pred_arr = pred_arr[pred_arr[:, 5] >= args.conf]

        gt_boxes = []
        for row in gt_arr:
            class_id = int(row[0])
            if class_id >= NUM_CLASSES or class_id < 0:
                continue
            gt_boxes.append(
                {
                    "class": class_id,
                    "x_center": float(row[1]),
                    "y_center": float(row[2]),
                    "width": float(row[3]),
                    "height": float(row[4]),
                    "conf": float(row[5]) if len(row) > 5 else 1.0,
                }
            )

        pred_boxes = []
        for row in pred_arr:
            class_id = int(row[0])
            if class_id >= NUM_CLASSES or class_id < 0:
                continue
            pred_boxes.append(
                {
                    "class": class_id,
                    "x_center": float(row[1]),
                    "y_center": float(row[2]),
                    "width": float(row[3]),
                    "height": float(row[4]),
                    "conf": float(row[5]),
                }
            )

        gt_mask = np.full((height, width), BACKGROUND_IDX, dtype=np.uint8)
        pred_mask = np.full((height, width), BACKGROUND_IDX, dtype=np.uint8)

        gt_boxes = sorted(
            gt_boxes, key=lambda b: (CLASS_PRIORITIES[b["class"]], b["conf"])
        )
        pred_boxes = sorted(
            pred_boxes, key=lambda b: (CLASS_PRIORITIES[b["class"]], b["conf"])
        )

        for box in gt_boxes:
            x1 = max(
                0,
                min(width, int(np.floor((box["x_center"] - box["width"] / 2) * width))),
            )
            y1 = max(
                0,
                min(
                    height,
                    int(np.floor((box["y_center"] - box["height"] / 2) * height)),
                ),
            )
            x2 = max(
                0,
                min(width, int(np.ceil((box["x_center"] + box["width"] / 2) * width))),
            )
            y2 = max(
                0,
                min(
                    height, int(np.ceil((box["y_center"] + box["height"] / 2) * height))
                ),
            )
            gt_mask[y1:y2, x1:x2] = box["class"]

        for box in pred_boxes:
            x1 = max(
                0,
                min(width, int(np.floor((box["x_center"] - box["width"] / 2) * width))),
            )
            y1 = max(
                0,
                min(
                    height,
                    int(np.floor((box["y_center"] - box["height"] / 2) * height)),
                ),
            )
            x2 = max(
                0,
                min(width, int(np.ceil((box["x_center"] + box["width"] / 2) * width))),
            )
            y2 = max(
                0,
                min(
                    height, int(np.ceil((box["y_center"] + box["height"] / 2) * height))
                ),
            )
            pred_mask[y1:y2, x1:x2] = box["class"]

        gray = np.array(
            Image.open(CHANNEL0 / f"{pred_stem}.png").convert("L"), dtype=np.uint8
        )
        black_mask = gray == 0
        gt_mask[black_mask] = BACKGROUND_IDX
        pred_mask[black_mask] = BACKGROUND_IDX

        n = NUM_CLASSES + 1
        flat = gt_mask.astype(np.int32).ravel() * n + pred_mask.astype(np.int32).ravel()
        counts_1d = np.bincount(flat, minlength=n * n)
        counts += counts_1d.reshape((n, n))

        gt_color = np.zeros((height, width, 3), dtype=np.uint8)
        pred_color = np.zeros((height, width, 3), dtype=np.uint8)

        for cls in range(NUM_CLASSES):
            gt_color[gt_mask == cls] = CLASS_COLORS[cls]
            pred_color[pred_mask == cls] = CLASS_COLORS[cls]

        tree_assigned_dir = args.assigned_pixels_dir / tree_name
        tree_assigned_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(tree_assigned_dir / f"{stem}_gt_assigned.png"), gt_color)
        cv2.imwrite(str(tree_assigned_dir / f"{stem}_pred_assigned.png"), pred_color)

    class_metrics = []
    for cls in range(NUM_CLASSES):
        tp = int(counts[cls, cls])
        fp = int(counts[:, cls].sum() - tp)
        fn = int(counts[cls, :].sum() - tp)

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            (2 * precision * recall) / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        IoU = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0

        class_metrics.append(
            {
                "class": cls,
                "name": YOLO_CLASS_NAMES[cls],
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "iou": IoU,
            }
        )

    total_tp = sum(item["tp"] for item in class_metrics)
    total_fp = sum(item["fp"] for item in class_metrics)
    total_fn = sum(item["fn"] for item in class_metrics)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    f1 = (
        (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
    )

    foreground_sum = int(counts[:NUM_CLASSES, :].sum())
    total_pixels = int(counts.sum())
    total_correct = int(np.trace(counts))
    foreground_correct = int(np.trace(counts[:NUM_CLASSES, :NUM_CLASSES]))
    pixel_accuracy = total_correct / total_pixels if total_pixels else 0.0
    foreground_accuracy = foreground_correct / foreground_sum if foreground_sum else 0.0
    miou = float(np.mean([item["iou"] for item in class_metrics]))

    print(f"RESULTS FOR TREE: {tree_name}")
    print(
        f"\n{'Class':<15} {'TP(px)':>12} {'FP(px)':>12} {'FN(px)':>12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'IoU':>10}"
    )

    for item in class_metrics:
        print(
            f"{item['name']:<15} {item['tp']:>12} {item['fp']:>12} {item['fn']:>12} "
            f"{item['precision']:>10.4f} {item['recall']:>10.4f} {item['f1']:>10.4f} {item['iou']:>10.4f}"
        )

    print(
        f"{'OVERALL':<15} {total_tp:>12} {total_fp:>12} {total_fn:>12} "
        f"{precision:>10.4f} {recall:>10.4f} {f1:>10.4f} {miou:>10.4f}"
    )
    print(f"\n{'Pixel Accuracy:':<20} {pixel_accuracy:.4f}")
    print(f"{'Foreground Acc.:':<20} {foreground_accuracy:.4f}")

    tree_summaries.append(
        {
            "tree": tree_name,
            "total_tp": int(total_tp),
            "total_fp": int(total_fp),
            "total_fn": int(total_fn),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "pixel_accuracy": float(pixel_accuracy),
            "foreground_accuracy": float(foreground_accuracy),
            "miou": float(miou),
            "total_pixels": int(total_pixels),
            "class_metrics": class_metrics,
        }
    )

    total_confusion_counts += counts

    labels = YOLO_CLASS_NAMES + ["background"]
    row_sums = counts.sum(axis=1, keepdims=True)
    matrix_to_plot = np.divide(
        counts, row_sums, out=np.zeros_like(counts, dtype=float), where=row_sums > 0
    ).T
    plot_counts = counts.T

    tree_conf_dir = args.output_root / "confusion_matrices" / tree_name
    tree_conf_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix_to_plot, cmap="Blues", vmin=0, vmax=1.0)
    ax.set_title(f"Confusion Matrix - {tree_name}")
    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)

    max_val = matrix_to_plot.max()
    for i, j in itertools.product(
        range(matrix_to_plot.shape[0]), range(matrix_to_plot.shape[1])
    ):
        value = matrix_to_plot[i, j]
        count = plot_counts[i, j]
        color = "white" if value > max_val * 0.5 else "#1a1a1a"
        ax.text(
            j,
            i,
            f"{value:.2f}\n({count})",
            ha="center",
            va="center",
            color=color,
            fontsize=9,
        )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Proportion", rotation=90)
    fig.tight_layout()
    fig.savefig(tree_conf_dir / "confusion_matrix_normalized.png", dpi=200)
    plt.close(fig)

total_tp = sum(item["total_tp"] for item in tree_summaries)
total_fp = sum(item["total_fp"] for item in tree_summaries)
total_fn = sum(item["total_fn"] for item in tree_summaries)

precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0

total_pixels = sum(item["total_pixels"] for item in tree_summaries)
pixel_accuracy = (
    sum(item["pixel_accuracy"] * item["total_pixels"] for item in tree_summaries)
    / total_pixels
    if total_pixels
    else 0.0
)
mean_miou = (
    float(np.mean([item["miou"] for item in tree_summaries])) if tree_summaries else 0.0
)

print("\n ====== OVERALL SUMMARY ======")
print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
print(f"Pixel Accuracy: {pixel_accuracy:.4f} | mIoU: {mean_miou:.4f}")

labels = YOLO_CLASS_NAMES + ["background"]
row_sums = total_confusion_counts.sum(axis=1, keepdims=True)
matrix_to_plot = np.divide(
    total_confusion_counts,
    row_sums,
    out=np.zeros_like(total_confusion_counts, dtype=float),
    where=row_sums > 0,
).T
plot_counts = total_confusion_counts.T

all_conf_dir = args.output_root / "confusion_matrices"
all_conf_dir.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(matrix_to_plot, cmap="Blues", vmin=0, vmax=1.0)
ax.set_title("Confusion Matrix")
ax.set_xlabel("True")
ax.set_ylabel("Predicted")
ax.set_xticks(np.arange(len(labels)))
ax.set_xticklabels(labels, rotation=90)
ax.set_yticks(np.arange(len(labels)))
ax.set_yticklabels(labels)

max_val = matrix_to_plot.max()
for i, j in itertools.product(
    range(matrix_to_plot.shape[0]), range(matrix_to_plot.shape[1])
):
    value = matrix_to_plot[i, j]
    count = plot_counts[i, j]
    color = "white" if value > max_val * 0.5 else "#1a1a1a"
    ax.text(
        j,
        i,
        f"{value:.2f}\n({count})",
        ha="center",
        va="center",
        color=color,
        fontsize=9,
    )

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Proportion", rotation=90)
fig.tight_layout()
fig.savefig(all_conf_dir / "confusion_matrix_normalized.png", dpi=200)
plt.close(fig)
