import laspy
import numpy as np
import cv2
import os
import json

# --- CONFIGURATION ---
FILE_NAME = "group_321_GP.las"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.join(SCRIPT_DIR, FILE_NAME)
BASE_OUTPUT_FOLDER = "tree_slices"
SLICE_HEIGHT = 0.20  
PIXEL_SIZE = 0.01    

tree_name = os.path.splitext(os.path.basename(FILE_PATH))[0]
tree_folder = os.path.join(BASE_OUTPUT_FOLDER, tree_name)
os.makedirs(tree_folder, exist_ok=True)

# 1. Load the Point Cloud
print(f"Loading {FILE_NAME}...")
las = laspy.read(FILE_PATH)
points = np.vstack((las.x, las.y, las.z)).T

# 2. CALCULATE GLOBAL BOUNDS (The "Canvas")
# This ensures every image is the same size
x_min_global, x_max_global = np.min(points[:, 0]), np.max(points[:, 0])
y_min_global, y_max_global = np.min(points[:, 1]), np.max(points[:, 1])
z_min, z_max = np.min(points[:, 2]), np.max(points[:, 2])

# Constant Image Dimensions
canvas_width_px = int((x_max_global - x_min_global) / PIXEL_SIZE) + 1
canvas_height_px = int((y_max_global - y_min_global) / PIXEL_SIZE) + 1

num_slices = int((z_max - z_min) / SLICE_HEIGHT)
metadata = {}

print(f"Canvas Size: {canvas_width_px}x{canvas_height_px} pixels")
print(f"Processing {num_slices} slices...")

# 3. Process Each Slice
for i in range(num_slices):
    z_low = z_min + (i * SLICE_HEIGHT)
    z_high = z_low + SLICE_HEIGHT
    
    mask = (points[:, 2] >= z_low) & (points[:, 2] < z_high)
    slice_points = points[mask]
    
    if len(slice_points) < 5:
        continue

    # Create 2D Density Histogram using GLOBAL ranges
    counts, _, _ = np.histogram2d(
        slice_points[:, 0], slice_points[:, 1], 
        bins=[canvas_width_px, canvas_height_px], 
        range=[[x_min_global, x_max_global], [y_min_global, y_max_global]]
    )
    
    # Normalize
    max_val = np.percentile(counts, 99) if np.max(counts) > 0 else 1
    raster = np.clip(counts * (255 / (max_val + 1e-6)), 0, 255).astype(np.uint8)
    
    # Save Image (Transposed for XY alignment)
    img_name = f"{tree_name}_slice_{i:03d}.png"
    cv2.imwrite(os.path.join(tree_folder, img_name), raster.T)
    
    # Metadata is now much simpler because origins are constant!
    metadata[img_name] = {
        "x_origin_global": float(x_min_global),
        "y_origin_global": float(y_min_global),
        "z_layer": float(z_low),
        "pixel_size": PIXEL_SIZE,
        "canvas_w": canvas_width_px,
        "canvas_h": canvas_height_px
    }

with open(os.path.join(tree_folder, "metadata.json"), "w") as f:
    json.dump(metadata, f, indent=4)

print(f"Done! All images are {canvas_width_px}x{canvas_height_px}.")
