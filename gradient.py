from pathlib import Path
import re
import sys

import numpy as np
from PIL import Image


# ============================================================
# Configuration
# ============================================================
BASE_DIR = Path(r"C:\Users\georg\Desktop\a MSc Forestry\Modern Methods in TLS and UAV\Final Project\YOLO_Extraction_1")

OBJ_NAMES_FILE = BASE_DIR / "obj.names"
TRAIN_FILE = BASE_DIR / "train.txt"
OBJ_TRAIN_DATA_DIR = BASE_DIR / "obj_train_data"
OUTPUT_DIR = BASE_DIR / "gradient_images"

# Circle size mode:
# "max_half"     -> radius = max(box_width, box_height) / 2
# "avg_half"     -> radius = (box_width + box_height) / 4
# "fixed_pixels" -> always use FIXED_RADIUS_PX
RADIUS_MODE = "max_half"
FIXED_RADIUS_PX = 10

# If True, only process images listed in train.txt
# If False, process every image in obj_train_data
USE_TRAIN_TXT = True

# Supported image extensions
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


# ============================================================
# Helpers
# ============================================================
def load_class_names(obj_names_file: Path) -> list[str]:
    if not obj_names_file.exists():
        raise FileNotFoundError(f"Could not find: {obj_names_file}")

    with obj_names_file.open("r", encoding="utf-8") as f:
        names = [line.strip() for line in f if line.strip()]

    if not names:
        raise ValueError("obj.names is empty.")

    return names


def find_trunk_class_index(class_names: list[str]) -> int:
    # Exact match first
    for i, name in enumerate(class_names):
        if name.strip().lower() == "trunk":
            return i

    # Fallback: accept "trunks"
    for i, name in enumerate(class_names):
        if name.strip().lower() == "trunks":
            return i

    raise ValueError(
        "Could not find class 'trunk' in obj.names.\n"
        f"Classes found: {class_names}"
    )


def read_train_image_paths(train_file: Path, obj_train_data_dir: Path) -> list[Path]:
    if not train_file.exists():
        raise FileNotFoundError(f"Could not find: {train_file}")

    image_paths = []

    with train_file.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            # Keep only the filename part, because train.txt may use paths like:
            # data/obj_train_data/group_321_GP_slice_000.png
            filename = Path(line.replace("\\", "/")).name
            candidate = obj_train_data_dir / filename

            if candidate.exists():
                image_paths.append(candidate)
            else:
                print(f"Warning: Image listed in train.txt not found: {candidate}")

    return image_paths


def find_all_images(obj_train_data_dir: Path) -> list[Path]:
    return sorted(
        [p for p in obj_train_data_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
    )


def label_file_for_image(image_path: Path) -> Path:
    return image_path.with_suffix(".txt")


def parse_yolo_label_file(label_file: Path, trunk_index: int, img_w: int, img_h: int):
    """
    Returns a list of trunk circles as:
    [(cx_px, cy_px, radius_px), ...]
    """
    circles = []

    if not label_file.exists():
        return circles

    with label_file.open("r", encoding="utf-8") as f:
        for line_number, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) != 5:
                print(f"Warning: Skipping malformed line in {label_file.name} (line {line_number}): {line}")
                continue

            try:
                class_idx = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                box_w = float(parts[3])
                box_h = float(parts[4])
            except ValueError:
                print(f"Warning: Skipping unreadable line in {label_file.name} (line {line_number}): {line}")
                continue

            if class_idx != trunk_index:
                continue

            cx_px = x_center * img_w
            cy_px = y_center * img_h
            bw_px = box_w * img_w
            bh_px = box_h * img_h

            if RADIUS_MODE == "max_half":
                radius_px = max(bw_px, bh_px) / 2.0
            elif RADIUS_MODE == "avg_half":
                radius_px = (bw_px + bh_px) / 4.0
            elif RADIUS_MODE == "fixed_pixels":
                radius_px = float(FIXED_RADIUS_PX)
            else:
                raise ValueError(f"Unknown RADIUS_MODE: {RADIUS_MODE}")

            circles.append((cx_px, cy_px, max(radius_px, 1.0)))

    return circles


def generate_gradient_image(img_w: int, img_h: int, circles: list[tuple[float, float, float]]) -> np.ndarray:
    """
    Creates a grayscale gradient image:
    - white circles at trunk locations
    - outside circles: brightness fades with distance from nearest trunk circle
    - farthest pixel becomes black
    """

    if not circles:
        # No trunks -> all black
        return np.zeros((img_h, img_w), dtype=np.uint8)

    yy, xx = np.indices((img_h, img_w), dtype=np.float32)

    min_distance = np.full((img_h, img_w), np.inf, dtype=np.float32)

    for cx, cy, radius in circles:
        dist_to_center = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        dist_to_circle = np.maximum(dist_to_center - radius, 0.0)
        min_distance = np.minimum(min_distance, dist_to_circle)

    max_dist = float(np.max(min_distance))

    if max_dist <= 0:
        # Entire image covered by circles
        gradient = np.full((img_h, img_w), 255, dtype=np.uint8)
    else:
        # Normalized distance: 0 = at trunk circle, 1 = furthest point
        r = min_distance / max_dist
        # Piecewise gradient:
        # - first 25% of distance uses 50% of total darkness drop
        # - remaining 75% uses the other 50%
        gradient_float = np.empty_like(r, dtype=np.float32)
        first_region = r <= 0.25
        second_region = r > 0.25
        # Region 1: 255 -> 127.5 over r = 0.00 -> 0.25
        gradient_float[first_region] = 255.0 - (r[first_region] / 0.25) * 127.5
        # Region 2: 127.5 -> 0 over r = 0.25 -> 1.00
        gradient_float[second_region] = 127.5 * (1.0 - (r[second_region] - 0.25) / 0.75)
        gradient = np.clip(gradient_float, 0, 255).astype(np.uint8)

    return gradient


def save_gradient_image(output_path: Path, gradient_array: np.ndarray):
    img = Image.fromarray(gradient_array, mode="L")
    img.save(output_path)


# ============================================================
# Main
# ============================================================
def main():
    if not BASE_DIR.exists():
        raise FileNotFoundError(f"Base directory does not exist: {BASE_DIR}")

    if not OBJ_TRAIN_DATA_DIR.exists():
        raise FileNotFoundError(f"Could not find folder: {OBJ_TRAIN_DATA_DIR}")

    OUTPUT_DIR.mkdir(exist_ok=True)

    class_names = load_class_names(OBJ_NAMES_FILE)
    trunk_index = find_trunk_class_index(class_names)

    print(f"Found trunk class at index: {trunk_index}")

    if USE_TRAIN_TXT:
        image_paths = read_train_image_paths(TRAIN_FILE, OBJ_TRAIN_DATA_DIR)
    else:
        image_paths = find_all_images(OBJ_TRAIN_DATA_DIR)

    if not image_paths:
        print("No images found to process.")
        return

    total = len(image_paths)
    processed = 0

    for i, image_path in enumerate(image_paths, start=1):
        try:
            with Image.open(image_path) as img:
                img_w, img_h = img.size

            label_file = label_file_for_image(image_path)
            circles = parse_yolo_label_file(label_file, trunk_index, img_w, img_h)
            gradient = generate_gradient_image(img_w, img_h, circles)

            output_path = OUTPUT_DIR / image_path.name
            save_gradient_image(output_path, gradient)

            print(f"[{i}/{total}] Saved: {output_path.name} | trunks found: {len(circles)}")
            processed += 1

        except Exception as e:
            print(f"[{i}/{total}] Error processing {image_path.name}: {e}")

    print()
    print(f"Done. Processed {processed} out of {total} images.")
    print(f"Output folder: {OUTPUT_DIR}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Fatal error: {exc}")
        sys.exit(1)