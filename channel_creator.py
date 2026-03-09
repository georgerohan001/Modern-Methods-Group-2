from pathlib import Path
import sys
import numpy as np
from PIL import Image, ImageDraw

# ======================================================
# Paths
# ======================================================

BASE_DIR = Path(r"C:\Users\georg\Desktop\a MSc Forestry\Modern Methods in TLS and UAV\Final Project\YOLO_Extraction_1")

OBJ_NAMES_FILE = BASE_DIR / "obj.names"
TRAIN_FILE = BASE_DIR / "train.txt"
OBJ_TRAIN_DATA = BASE_DIR / "obj_train_data"

INDEX_OUTPUT = BASE_DIR / "index_gray_images"
LABEL_OUTPUT = BASE_DIR / "label_polygon_images"
GRADIENT_OUTPUT = BASE_DIR / "gradient_images"

INDEX_OUTPUT.mkdir(exist_ok=True)
LABEL_OUTPUT.mkdir(exist_ok=True)
GRADIENT_OUTPUT.mkdir(exist_ok=True)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# ======================================================
# Config
# ======================================================

USE_TRAIN_TXT = True

LABEL_COLORS = {
    0: 255,
    1: 205,
    2: 155,
    3: 105,
    4: 55
}

RADIUS_MODE = "max_half"
FIXED_RADIUS_PX = 10


# ======================================================
# Helper functions
# ======================================================

def load_class_names(obj_names_file):
    with open(obj_names_file, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def find_trunk_class_index(class_names):
    for i, name in enumerate(class_names):
        if name.lower() == "trunk":
            return i
    for i, name in enumerate(class_names):
        if name.lower() == "trunks":
            return i
    raise ValueError("Could not find class 'trunk' in obj.names")


def get_image_list():

    if USE_TRAIN_TXT:

        image_paths = []

        with open(TRAIN_FILE, "r", encoding="utf-8") as f:
            for line in f:
                filename = Path(line.strip()).name
                candidate = OBJ_TRAIN_DATA / filename
                if candidate.exists():
                    image_paths.append(candidate)

        return sorted(image_paths)

    else:

        return sorted([
            p for p in OBJ_TRAIN_DATA.iterdir()
            if p.suffix.lower() in IMAGE_EXTENSIONS
        ])


def parse_yolo_labels(label_file, width, height, trunk_index):

    circles = []
    boxes = []

    if not label_file.exists():
        return circles, boxes

    with open(label_file, "r") as f:

        for line in f:

            parts = line.strip().split()

            if len(parts) != 5:
                continue

            label = int(parts[0])
            cx = float(parts[1])
            cy = float(parts[2])
            bw = float(parts[3])
            bh = float(parts[4])

            cx *= width
            cy *= height
            bw *= width
            bh *= height

            x1 = cx - bw/2
            y1 = cy - bh/2
            x2 = cx + bw/2
            y2 = cy + bh/2

            boxes.append((label, x1, y1, x2, y2))

            if label == trunk_index:

                if RADIUS_MODE == "max_half":
                    radius = max(bw, bh)/2
                elif RADIUS_MODE == "avg_half":
                    radius = (bw + bh)/4
                else:
                    radius = FIXED_RADIUS_PX

                circles.append((cx, cy, max(radius,1)))

    return circles, boxes


def generate_gradient_image(width, height, circles):

    if not circles:
        return np.zeros((height, width), dtype=np.uint8)

    yy, xx = np.indices((height, width), dtype=np.float32)

    min_distance = np.full((height, width), np.inf)

    for cx, cy, r in circles:

        dist_center = np.sqrt((xx-cx)**2 + (yy-cy)**2)
        dist_circle = np.maximum(dist_center - r, 0)

        min_distance = np.minimum(min_distance, dist_circle)

    max_dist = np.max(min_distance)

    if max_dist <= 0:
        return np.full((height,width),255,dtype=np.uint8)

    r = min_distance / max_dist

    gradient = np.empty_like(r)

    region1 = r <= 0.25
    region2 = r > 0.25

    gradient[region1] = 255 - (r[region1]/0.25)*127.5
    gradient[region2] = 127.5*(1-(r[region2]-0.25)/0.75)

    return np.clip(gradient,0,255).astype(np.uint8)


# ======================================================
# Main
# ======================================================

def main():

    class_names = load_class_names(OBJ_NAMES_FILE)
    trunk_index = find_trunk_class_index(class_names)

    images = get_image_list()
    n = len(images)

    if n == 0:
        print("No images found")
        return

    print(f"Processing {n} images")

    for i, img_path in enumerate(images):

        img = Image.open(img_path)
        width, height = img.size

        label_file = img_path.with_suffix(".txt")

        circles, boxes = parse_yolo_labels(label_file, width, height, trunk_index)

        # ==========================================
        # 1) Index grayscale image
        # ==========================================

        if n == 1:
            gray_value = 0
        else:
            gray_value = int(round((i/(n-1))*255))

        index_arr = np.full((height,width), gray_value, dtype=np.uint8)

        Image.fromarray(index_arr).save(INDEX_OUTPUT / img_path.name)

        # ==========================================
        # 2) Label mask image
        # ==========================================

        mask = Image.new("L",(width,height),0)
        draw = ImageDraw.Draw(mask)

        for label,x1,y1,x2,y2 in boxes:
            color = LABEL_COLORS.get(label,0)
            draw.rectangle([x1,y1,x2,y2], fill=color)

        mask.save(LABEL_OUTPUT / img_path.name)

        # ==========================================
        # 3) Gradient trunk image
        # ==========================================

        gradient = generate_gradient_image(width,height,circles)

        Image.fromarray(gradient).save(GRADIENT_OUTPUT / img_path.name)

        print(f"[{i+1}/{n}] processed {img_path.name}")

    print("\nDone.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)