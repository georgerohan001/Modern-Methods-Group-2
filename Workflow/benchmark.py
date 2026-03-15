from __future__ import annotations

import argparse
import itertools
import re
import shutil
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import laspy
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from ultralytics import YOLO

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "Model" / "best.pt"
INPUT_DIR = BASE_DIR / "INPUT"
IMAGES_DIR = BASE_DIR / "Images"
CHANNELS_DIR = BASE_DIR / "Channels"
TEST_DIR = BASE_DIR / "TEST"
PREDICTED_DIR = BASE_DIR / "PREDICTED"
OUTPUT_DIR = BASE_DIR / "OUTPUT" / "benchmark"
ASSIGNED_DIR = BASE_DIR / "Assigned"

CHANNEL0 = CHANNELS_DIR / "channel0"
CHANNEL1 = CHANNELS_DIR / "channel1"
CHANNEL2 = CHANNELS_DIR / "channel2"
CHANNEL3 = CHANNELS_DIR / "channel3"

IMAGE_EXTENSIONS = {".tif", ".png"}
SLICE_HEIGHT = 0.20
PIXEL_SIZE = 0.01
IOU_THRESHOLD = 0.70
IMAGE_SIZE = 640
DEVICE = "cpu"


# =============================================================================
# CLASS CONFIGURATION
# =============================================================================


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
CLASS_COLORS = {i: cfg.color for i, cfg in CLASSES.items()}
CLASS_PRIORITIES = {i: cfg.priority for i, cfg in CLASSES.items()}
CLASS_NAMES = [CLASSES[i].name for i in range(NUM_CLASSES)]


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Unified YOLO Benchmark")
    parser.add_argument("--conf", type=float, default=0.10, help="Confidence threshold")
    parser.add_argument("--tree", action="append", default=None, help="Filter trees")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--assigned-pixels-dir", type=Path, default=ASSIGNED_DIR)
    return parser.parse_args()


def ensure_dirs(args: argparse.Namespace) -> None:
    """Create necessary directories."""
    CHANNEL0.mkdir(parents=True, exist_ok=True)
    CHANNEL1.mkdir(parents=True, exist_ok=True)
    CHANNEL2.mkdir(parents=True, exist_ok=True)
    CHANNEL3.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTED_DIR.mkdir(parents=True, exist_ok=True)
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "confusion_matrices").mkdir(parents=True, exist_ok=True)
    args.assigned_pixels_dir.mkdir(parents=True, exist_ok=True)


def load_model() -> YOLO:
    """Load YOLO model and return class names."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    model = YOLO(str(MODEL_PATH))
    return model


def get_model_class_names(model: YOLO) -> list[str]:
    """Extract class names from YOLO model."""
    names_attr = getattr(model, "names", None)
    if isinstance(names_attr, dict):
        return [names_attr[i] for i in sorted(names_attr.keys())]
    return [str(x) for x in names_attr] if names_attr else CLASS_NAMES


# =============================================================================
# IMAGE GENERATION FUNCTIONS
# =============================================================================


def count_expected_images() -> int:
    """Calculate expected number of images from LAS files."""
    las_files = sorted(INPUT_DIR.glob("*.las"))
    expected = 0
    for las_file in las_files:
        try:
            las = laspy.read(las_file)
            points = np.column_stack((las.x, las.y, las.z))
            if points.size == 0:
                continue
            z_min, z_max = np.min(points[:, 2]), np.max(points[:, 2])
            expected += int((z_max - z_min) / SLICE_HEIGHT)
        except Exception:
            continue
    return expected


def check_existing_images() -> int:
    """Count existing TIFF files in Images directory."""
    return len([p for p in IMAGES_DIR.glob("*.tif") if p.is_file()])


def generate_images_from_las(metadata_store: dict) -> None:
    """Generate multi-channel TIFF images from LAS files."""
    print("\n[1/4] Generating images from LAS files...")

    las_files = sorted(INPUT_DIR.glob("*.las"))
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

            max_val = np.percentile(counts, 99) if np.max(counts) > 0 else 1
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

    build_multichannel_images()
    print(f"   Generated {check_existing_images()} images")


def build_multichannel_images() -> None:
    """Generate all 4 channels and save as RGBA TIFF in one pass."""
    channel0_files = sorted(CHANNEL0.glob("*.png"))

    # Group by tree stem (e.g., "tree_slice_001.png" -> "tree_slice")
    grouped = defaultdict(list)
    for p in channel0_files:
        stem = p.stem.rsplit("_", 1)[0]
        grouped[stem].append(p)

    for tree_stem, paths in grouped.items():
        # Sort by slice number
        paths.sort(key=lambda p: int(p.stem.split("_")[-1]))

        for i, ch0_path in enumerate(paths):
            # Load channel 0
            ch0 = np.array(Image.open(ch0_path).convert("L"))
            h, w = ch0.shape

            # Channel 1: previous slice (or same if first)
            ch1_path = paths[max(i - 1, 0)]
            ch1 = np.array(Image.open(ch1_path).convert("L"))

            # Channel 2: two slices back (or same)
            ch2_path = paths[max(i - 2, 0)]
            ch2 = np.array(Image.open(ch2_path).convert("L"))

            # Channel 3: gradient position (0-255)
            gray_val = int(255 * i / max(len(paths) - 1, 1))
            ch3 = np.full((h, w), gray_val, dtype=np.uint8)

            # Stack and save as RGBA TIFF
            stacked = np.stack([ch0, ch1, ch2, ch3], axis=2)
            Image.fromarray(stacked).save(
                IMAGES_DIR / f"{ch0_path.stem}.tif", compression="tiff_deflate"
            )


# =============================================================================
# YOLO INFERENCE FUNCTIONS
# =============================================================================


def run_yolo_inference(model: YOLO, conf: float) -> None:
    """Run YOLO inference on images and cache predictions."""
    print("\n[2/4] Running YOLO inference...")

    images = sorted(IMAGES_DIR.glob("*.tif"))
    if not images:
        raise RuntimeError("No images found for inference")

    image_stems = {p.stem for p in images}
    pred_stems = {p.stem for p in PREDICTED_DIR.glob("*.txt")}

    if image_stems.issubset(pred_stems):
        print("   Reusing existing predictions")
        return

    model.predict(
        source=str(IMAGES_DIR),
        imgsz=IMAGE_SIZE,
        conf=conf,
        iou=IOU_THRESHOLD,
        save=False,
        save_txt=True,
        save_conf=True,
        project=str(PREDICTED_DIR.parent),
        name=PREDICTED_DIR.name,
        exist_ok=True,
        device=DEVICE,
    )

    # Ensure all prediction files exist (touch missing ones)
    for stem in image_stems:
        (PREDICTED_DIR / f"{stem}.txt").touch()

    print(f"   Generated {sum(1 for _ in PREDICTED_DIR.glob('*.txt'))} predictions")


# =============================================================================
# BOUNDING BOX METRICS FUNCTIONS
# =============================================================================


def prepare_temp_dataset(tree_dirs: list[Path]) -> Path:
    """Prepare temporary dataset for YOLO validation."""
    dataset_root = OUTPUT_DIR / "tmp_dataset"
    images_out = dataset_root / "images" / "test"
    labels_out = dataset_root / "labels" / "test"

    shutil.rmtree(dataset_root, ignore_errors=True)
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    image_index = {p.stem: p for p in IMAGES_DIR.glob("*.tif")}

    for tree_dir in tree_dirs:
        obj_data = tree_dir / "obj_train_data"
        if not obj_data.exists():
            continue

        for gt_file in sorted(obj_data.glob("*.txt")):
            stem = _match_image_stem(gt_file.stem, image_index)
            if stem is None:
                continue

            image_path = image_index[stem]
            out_stem = f"{tree_dir.name}__{stem}"

            # Copy image
            out_image = images_out / f"{out_stem}.tif"
            if not out_image.exists():
                shutil.copy2(image_path, out_image)

            # Copy labels as-is (no remapping)
            if gt_file.exists() and gt_file.stat().st_size > 0:
                shutil.copy2(gt_file, labels_out / f"{out_stem}.txt")
            else:
                (labels_out / f"{out_stem}.txt").touch()

    return dataset_root


def _match_image_stem(gt_stem: str, image_index: dict[str, Path]) -> str | None:
    """Match ground truth stem to image stem."""
    candidates = [gt_stem, f"tree_{gt_stem}"]
    if gt_stem.startswith("tree_"):
        candidates.append(gt_stem[5:])

    for c in candidates:
        if c in image_index:
            return c
    return None


def create_data_yaml(dataset_root: Path, class_names: list[str]) -> Path:
    """Create data.yaml configuration file."""
    data_yaml = dataset_root / "data.yaml"

    lines = [
        f"path: {dataset_root.as_posix()}",
        "train: images/test",
        "val: images/test",
        "test: images/test",
        f"nc: {len(class_names)}",
        "names:",
    ]
    for i, name in enumerate(class_names):
        escaped = name.replace('"', '\\"')
        lines.append(f'  {i}: "{escaped}"')

    data_yaml.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return data_yaml


def run_bbox_validation(
    model: YOLO, dataset_root: Path, class_names: list[str], conf: float
) -> dict[str, float]:
    """Run YOLO validation and extract bounding box metrics."""
    print("\n[3/4] Calculating bounding-box metrics...")

    data_yaml = create_data_yaml(dataset_root, class_names)

    results = model.val(
        data=str(data_yaml),
        imgsz=IMAGE_SIZE,
        conf=conf,
        iou=IOU_THRESHOLD,
        plots=False,
        device=DEVICE,
        verbose=False,
    )

    box = getattr(results, "box", None)
    if box is None:
        raise RuntimeError("YOLO validation did not return box metrics")

    mp = getattr(box, "mp", 0.0)
    mr = getattr(box, "mr", 0.0)
    f1 = 2 * mp * mr / (mp + mr) if (mp + mr) > 0 else 0.0

    metrics = {
        "map50": float(getattr(box, "map50", 0.0)),
        "map50_95": float(getattr(box, "map", 0.0)),
        "map75": float(getattr(box, "map75", 0.0)),
        "mp": float(mp),
        "mr": float(mr),
        "f1": float(f1),
    }

    print(f"   mAP50: {metrics['map50']:.4f}")
    print(f"   mAP50-95: {metrics['map50_95']:.4f}")
    print(f"   mAP75: {metrics['map75']:.4f}")
    print(f"   Precision: {metrics['mp']:.4f}")
    print(f"   Recall: {metrics['mr']:.4f}")
    print(f"   F1: {metrics['f1']:.4f}")

    return metrics


# =============================================================================
# PIXEL-LEVEL METRICS FUNCTIONS
# =============================================================================


def load_boxes_from_file(file_path: Path, num_classes: int) -> list[dict]:
    """Load bounding boxes from YOLO format text file."""
    if not file_path.exists() or file_path.stat().st_size == 0:
        return []

    try:
        arr = np.loadtxt(file_path, dtype=float)
    except Exception:
        return []

    if arr.size == 0:
        return []

    arr = np.atleast_2d(arr)
    if arr.shape[1] < 5:
        return []

    boxes = []
    for row in arr:
        class_id = int(row[0])
        if class_id < 0 or class_id >= num_classes:
            continue

        boxes.append(
            {
                "class": class_id,
                "x_center": float(row[1]),
                "y_center": float(row[2]),
                "width": float(row[3]),
                "height": float(row[4]),
                "conf": float(row[5]) if len(row) > 5 else 1.0,
            }
        )

    return boxes


def render_boxes_to_mask(
    boxes: list[dict], width: int, height: int, priorities: dict
) -> np.ndarray:
    """Render bounding boxes to a pixel mask."""
    mask = np.full((height, width), BACKGROUND_IDX, dtype=np.uint8)

    sorted_boxes = sorted(boxes, key=lambda b: priorities.get(b["class"], 0))

    for box in sorted_boxes:
        x1 = max(
            0, min(width, int(np.floor((box["x_center"] - box["width"] / 2) * width)))
        )
        y1 = max(
            0,
            min(height, int(np.floor((box["y_center"] - box["height"] / 2) * height))),
        )
        x2 = max(
            0, min(width, int(np.ceil((box["x_center"] + box["width"] / 2) * width)))
        )
        y2 = max(
            0, min(height, int(np.ceil((box["y_center"] + box["height"] / 2) * height)))
        )

        mask[y1:y2, x1:x2] = box["class"]

    return mask


def calculate_confusion_counts(
    gt_mask: np.ndarray, pred_mask: np.ndarray, num_classes: int
) -> np.ndarray:
    """Calculate confusion matrix counts from masks."""
    n = num_classes + 1  # Include background
    flat = gt_mask.astype(np.int32).ravel() * n + pred_mask.astype(np.int32).ravel()
    counts_1d = np.bincount(flat, minlength=n * n)
    return counts_1d.reshape((n, n))


def calculate_per_class_metrics(
    counts: np.ndarray, num_classes: int
) -> list[dict[str, Any]]:
    """Calculate per-class metrics from confusion counts."""
    metrics = []

    for cls in range(num_classes):
        tp = int(counts[cls, cls])
        fp = int(counts[:, cls].sum() - tp)
        fn = int(counts[cls, :].sum() - tp)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            (2 * precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

        metrics.append(
            {
                "class": cls,
                "name": CLASS_NAMES[cls],
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "iou": iou,
            }
        )

    return metrics


def calculate_overall_metrics(
    class_metrics: list[dict], counts: np.ndarray, num_classes: int
) -> dict[str, Any]:
    """Calculate overall metrics from per-class metrics."""
    total_tp = sum(m["tp"] for m in class_metrics)
    total_fp = sum(m["fp"] for m in class_metrics)
    total_fn = sum(m["fn"] for m in class_metrics)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = (
        (2 * precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    total_pixels = int(counts.sum())
    total_correct = int(np.trace(counts))
    pixel_accuracy = total_correct / total_pixels if total_pixels > 0 else 0.0

    foreground_sum = int(counts[:num_classes, :].sum())
    foreground_correct = int(np.trace(counts[:num_classes, :num_classes]))
    foreground_accuracy = (
        foreground_correct / foreground_sum if foreground_sum > 0 else 0.0
    )

    miou = float(np.mean([m["iou"] for m in class_metrics]))

    return {
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pixel_accuracy": pixel_accuracy,
        "foreground_accuracy": foreground_accuracy,
        "miou": miou,
        "total_pixels": total_pixels,
    }


def save_assigned_pixels(
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    stem: str,
    tree_name: str,
    assigned_dir: Path,
) -> None:
    """Save assigned pixel visualization images."""
    height, width = gt_mask.shape

    gt_color = np.zeros((height, width, 3), dtype=np.uint8)
    pred_color = np.zeros((height, width, 3), dtype=np.uint8)

    for cls in range(NUM_CLASSES):
        gt_color[gt_mask == cls] = CLASS_COLORS[cls]
        pred_color[pred_mask == cls] = CLASS_COLORS[cls]

    tree_assigned_dir = assigned_dir / tree_name
    tree_assigned_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(tree_assigned_dir / f"{stem}_gt_assigned.png"), gt_color)
    cv2.imwrite(str(tree_assigned_dir / f"{stem}_pred_assigned.png"), pred_color)


def plot_confusion_matrix(
    counts: np.ndarray,
    labels: list[str],
    title: str,
    output_path: Path,
) -> None:
    """Plot and save confusion matrix."""
    row_sums = counts.sum(axis=1, keepdims=True)
    matrix = np.divide(
        counts, row_sums, out=np.zeros_like(counts, dtype=float), where=row_sums > 0
    ).T

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1.0)

    ax.set_title(title)
    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)

    max_val = matrix.max()
    for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):
        value = matrix[i, j]
        color = "white" if value > max_val * 0.5 else "#1a1a1a"
        ax.text(j, i, f"{value:.2f}", ha="center", va="center", color=color, fontsize=9)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Proportion", rotation=90)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def process_tree_pixel_metrics(
    tree_dir: Path,
    conf: float,
    assigned_dir: Path,
    conf_dir: Path,
) -> tuple[dict, np.ndarray]:
    """Process pixel metrics for a single tree."""
    tree_name = tree_dir.name
    gt_dir = tree_dir / "obj_train_data"

    if not gt_dir.exists():
        raise FileNotFoundError(f"Ground truth directory not found: {gt_dir}")

    counts = np.zeros((NUM_CLASSES + 1, NUM_CLASSES + 1), dtype=np.int64)

    for gt_file in sorted(gt_dir.glob("*.txt")):
        stem = gt_file.stem
        pred_stem = stem[5:] if stem.startswith("tree_") else stem

        # Find prediction file
        pred_file = PREDICTED_DIR / f"{pred_stem}.txt"
        if not pred_file.exists():
            alt = PREDICTED_DIR / "labels" / f"{pred_stem}.txt"
            if alt.exists():
                pred_file = alt

        # Get image dimensions from channel0
        channel0_img = CHANNEL0 / f"{pred_stem}.png"
        if not channel0_img.exists():
            continue

        with Image.open(channel0_img) as img:
            width, height = img.size

        # Load boxes
        gt_boxes = load_boxes_from_file(gt_file, NUM_CLASSES)
        pred_boxes = load_boxes_from_file(pred_file, NUM_CLASSES)
        pred_boxes = [b for b in pred_boxes if b["conf"] >= conf]

        # Render masks
        gt_mask = render_boxes_to_mask(gt_boxes, width, height, CLASS_PRIORITIES)
        pred_mask = render_boxes_to_mask(pred_boxes, width, height, CLASS_PRIORITIES)

        # Apply black mask from channel0
        gray = np.array(Image.open(channel0_img).convert("L"), dtype=np.uint8)
        black_mask = gray == 0
        gt_mask[black_mask] = BACKGROUND_IDX
        pred_mask[black_mask] = BACKGROUND_IDX

        # Update confusion counts
        counts += calculate_confusion_counts(gt_mask, pred_mask, NUM_CLASSES)

        # Save assigned pixels
        save_assigned_pixels(gt_mask, pred_mask, stem, tree_name, assigned_dir)

    # Calculate metrics
    class_metrics = calculate_per_class_metrics(counts, NUM_CLASSES)
    overall = calculate_overall_metrics(class_metrics, counts, NUM_CLASSES)

    # Plot confusion matrix
    labels = CLASS_NAMES + ["background"]
    tree_conf_dir = conf_dir / tree_name
    tree_conf_dir.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix(
        counts,
        labels,
        f"Confusion Matrix - {tree_name}",
        tree_conf_dir / "confusion_matrix.png",
    )

    return {
        "tree": tree_name,
        "class_metrics": class_metrics,
        **overall,
    }, counts


def calculate_all_pixel_metrics(
    tree_dirs: list[Path],
    conf: float,
    assigned_dir: Path,
    output_root: Path,
) -> tuple[list[dict], np.ndarray]:
    """Calculate pixel metrics for all trees."""
    print("\n[4/4] Calculating pixel-level metrics...")

    conf_dir = output_root / "confusion_matrices"
    conf_dir.mkdir(parents=True, exist_ok=True)

    tree_summaries = []
    total_counts = np.zeros((NUM_CLASSES + 1, NUM_CLASSES + 1), dtype=np.int64)

    for tree_dir in tree_dirs:
        print(f"\n>>> Tree: {tree_dir.name}")

        try:
            summary, counts = process_tree_pixel_metrics(
                tree_dir, conf, assigned_dir, conf_dir
            )
            tree_summaries.append(summary)
            total_counts += counts

            # Print tree results
            _print_tree_results(summary)

        except Exception as e:
            print(f"   Error processing tree: {e}")
            continue

    return tree_summaries, total_counts


def _print_tree_results(summary: dict) -> None:
    """Print formatted results for a single tree."""
    print(f"\nRESULTS FOR TREE: {summary['tree']}")
    print(
        f"{'Class':<15} {'TP(px)':>12} {'FP(px)':>12} {'FN(px)':>12} "
        f"{'Precision':>10} {'Recall':>10} {'F1':>10} {'IoU':>10}"
    )

    for item in summary["class_metrics"]:
        print(
            f"{item['name']:<15} {item['tp']:>12} {item['fp']:>12} {item['fn']:>12} "
            f"{item['precision']:>10.4f} {item['recall']:>10.4f} {item['f1']:>10.4f} "
            f"{item['iou']:>10.4f}"
        )

    print(
        f"{'OVERALL':<15} {summary['total_tp']:>12} {summary['total_fp']:>12} "
        f"{summary['total_fn']:>12} {summary['precision']:>10.4f} {summary['recall']:>10.4f} "
        f"{summary['f1']:>10.4f} {summary['miou']:>10.4f}"
    )
    print(f"\n{'Pixel Accuracy:':<20} {summary['pixel_accuracy']:.4f}")
    print(f"{'Foreground Acc.:':<20} {summary['foreground_accuracy']:.4f}")


def print_final_summary(
    tree_summaries: list[dict],
    bbox_metrics: dict[str, float],
    total_counts: np.ndarray,
) -> None:
    """Print final overall summary."""
    if not tree_summaries:
        print("\nNo trees processed successfully.")
        return

    # Calculate overall pixel metrics
    total_tp = sum(s["total_tp"] for s in tree_summaries)
    total_fp = sum(s["total_fp"] for s in tree_summaries)
    total_fn = sum(s["total_fn"] for s in tree_summaries)
    total_pixels = sum(s["total_pixels"] for s in tree_summaries)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = (
        (2 * precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    pixel_acc = (
        sum(s["pixel_accuracy"] * s["total_pixels"] for s in tree_summaries)
        / total_pixels
        if total_pixels > 0
        else 0.0
    )
    miou = float(np.mean([s["miou"] for s in tree_summaries]))

    print("\n====== OVERALL SUMMARY ======")
    print(f"mAP50:          {bbox_metrics['map50']:.4f}")
    print(f"mAP50-95:       {bbox_metrics['map50_95']:.4f}")
    print(f"mIoU:           {miou:.4f}")
    print(f"Pixel Accuracy: {pixel_acc:.4f}")
    print(f"Precision:      {precision:.4f}")
    print(f"Recall:         {recall:.4f}")
    print(f"F1:             {f1:.4f}")


# =============================================================================
# CLEANUP FUNCTIONS
# =============================================================================


def cleanup(dataset_root: Path) -> None:
    """Clean up temporary files."""
    print("\n[5/5] Cleaning up temporary files...")
    if dataset_root.exists():
        shutil.rmtree(dataset_root, ignore_errors=True)
    print("   Cleanup complete")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================


def main() -> None:
    """Main benchmark execution."""
    args = parse_args()
    ensure_dirs(args)

    # Get tree directories
    tree_dirs = [
        d
        for d in sorted(TEST_DIR.iterdir())
        if d.is_dir()
        and (d / "obj.names").exists()
        and (not args.tree or d.name in args.tree)
    ]

    if not tree_dirs:
        raise RuntimeError(f"No tree directories found in {TEST_DIR}")

    print(f"Found {len(tree_dirs)} tree(s): {[d.name for d in tree_dirs]}")

    # Check/generate images
    expected = count_expected_images()
    existing = check_existing_images()

    metadata_store: dict = {}
    if expected > 0 and existing >= expected:
        print(f"Reusing {existing} existing images")
    else:
        generate_images_from_las(metadata_store)

    # Load model
    model = load_model()
    class_names = get_model_class_names(model)
    print(f"Model classes: {class_names}")

    # Run inference
    run_yolo_inference(model, args.conf)

    # Calculate bbox metrics
    dataset_root = prepare_temp_dataset(tree_dirs)
    bbox_metrics = run_bbox_validation(model, dataset_root, class_names, args.conf)

    # Calculate pixel metrics
    tree_summaries, total_counts = calculate_all_pixel_metrics(
        tree_dirs, args.conf, args.assigned_pixels_dir, args.output_root
    )

    # Print final summary (with updated mAP50 values)
    print_final_summary(tree_summaries, bbox_metrics, total_counts)

    # Plot overall confusion matrix
    if len(tree_summaries) > 1:
        labels = CLASS_NAMES + ["background"]
        plot_confusion_matrix(
            total_counts,
            labels,
            "Overall Confusion Matrix",
            args.output_root / "confusion_matrices" / "overall_confusion_matrix.png",
        )

    # Cleanup
    cleanup(dataset_root)

    print("\n✅ Benchmark complete!")


if __name__ == "__main__":
    main()
