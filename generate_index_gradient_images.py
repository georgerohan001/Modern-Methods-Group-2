from pathlib import Path
from PIL import Image
import numpy as np

# ======================================================
# Paths
# ======================================================

BASE_DIR = Path(r"C:\Users\georg\Desktop\a MSc Forestry\Modern Methods in TLS and UAV\Final Project\temp")

OBJ_TRAIN_DATA = BASE_DIR / "obj_train_data"
OUTPUT_DIR = BASE_DIR / "index_gray_images"

OUTPUT_DIR.mkdir(exist_ok=True)

# ======================================================
# Image extensions
# ======================================================

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# ======================================================
# Get image list
# ======================================================

images = sorted([p for p in OBJ_TRAIN_DATA.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS])

n = len(images)

if n == 0:
    print("No images found.")
    exit()

print(f"Found {n} images.")

# ======================================================
# Generate images
# ======================================================

for i, img_path in enumerate(images):

    # Load original image to get size
    img = Image.open(img_path)
    width, height = img.size

    # Compute grayscale value evenly from 0 → 255
    if n == 1:
        gray_value = 0
    else:
        gray_value = int(round((i / (n - 1)) * 255))

    # Create flat grayscale image
    arr = np.full((height, width), gray_value, dtype=np.uint8)
    new_img = Image.fromarray(arr, mode="L")

    # Save with same filename
    output_path = OUTPUT_DIR / img_path.name
    new_img.save(output_path)

    print(f"{img_path.name} -> gray value {gray_value}")

print("\nDone.")
print(f"Images saved in: {OUTPUT_DIR}")