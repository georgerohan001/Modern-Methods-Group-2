import laspy
import numpy as np
import cv2
import os
import json

# --- CONFIGURATION ---
INPUT_DIRECTORY = r"C:\Users\georg\Desktop\a MSc Forestry\Modern Methods in TLS and UAV\Final Project\Workflow\INPUT"   # Folder containing .las files
BASE_OUTPUT_FOLDER = "tree_slices"
SLICE_HEIGHT = 0.20
PIXEL_SIZE = 0.01

# -------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(SCRIPT_DIR, INPUT_DIRECTORY)

if not os.path.isdir(INPUT_PATH):
    raise ValueError(f"{INPUT_DIRECTORY} is not a valid directory.")

# Create output folder named same as input directory
output_root = os.path.join(BASE_OUTPUT_FOLDER, os.path.basename(INPUT_PATH))
os.makedirs(output_root, exist_ok=True)

# Collect metadata for ALL trees together
all_metadata = {}

# Get all LAS files
las_files = [f for f in os.listdir(INPUT_PATH) if f.lower().endswith(".las")]

print(f"Found {len(las_files)} LAS files in '{INPUT_DIRECTORY}'")

for FILE_NAME in las_files:
    FILE_PATH = os.path.join(INPUT_PATH, FILE_NAME)
    tree_name = os.path.splitext(FILE_NAME)[0]

    print(f"\nProcessing {FILE_NAME}...")

    # Load Point Cloud
    las = laspy.read(FILE_PATH)
    points = np.vstack((las.x, las.y, las.z)).T

    # Global bounds per tree
    x_min_global, x_max_global = np.min(points[:, 0]), np.max(points[:, 0])
    y_min_global, y_max_global = np.min(points[:, 1]), np.max(points[:, 1])
    z_min, z_max = np.min(points[:, 2]), np.max(points[:, 2])

    canvas_width_px = int((x_max_global - x_min_global) / PIXEL_SIZE) + 1
    canvas_height_px = int((y_max_global - y_min_global) / PIXEL_SIZE) + 1
    num_slices = int((z_max - z_min) / SLICE_HEIGHT)

    print(f"Canvas Size: {canvas_width_px}x{canvas_height_px} px")
    print(f"Processing {num_slices} slices...")

    for i in range(num_slices):
        z_low = z_min + (i * SLICE_HEIGHT)
        z_high = z_low + SLICE_HEIGHT

        mask = (points[:, 2] >= z_low) & (points[:, 2] < z_high)
        slice_points = points[mask]

        if len(slice_points) < 5:
            continue

        counts, _, _ = np.histogram2d(
            slice_points[:, 0],
            slice_points[:, 1],
            bins=[canvas_width_px, canvas_height_px],
            range=[[x_min_global, x_max_global],
                   [y_min_global, y_max_global]]
        )

        max_val = np.percentile(counts, 99) if np.max(counts) > 0 else 1
        raster = np.clip(counts * (255 / (max_val + 1e-6)), 0, 255).astype(np.uint8)

        # Image name includes tree name to avoid collision
        img_name = f"{tree_name}_slice_{i:03d}.png"
        cv2.imwrite(os.path.join(output_root, img_name), raster.T)

        # Store metadata
        all_metadata[img_name] = {
            "tree_name": tree_name,
            "x_origin_global": float(x_min_global),
            "y_origin_global": float(y_min_global),
            "z_layer": float(z_low),
            "pixel_size": PIXEL_SIZE,
            "canvas_w": canvas_width_px,
            "canvas_h": canvas_height_px
        }

    print(f"Finished {FILE_NAME}")

# Save ONE metadata file for everything
with open(os.path.join(output_root, "metadata.json"), "w") as f:
    json.dump(all_metadata, f, indent=4)

print("\n✅ All LAS files processed.")
print(f"All slices saved in: {output_root}")