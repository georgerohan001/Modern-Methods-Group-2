import laspy
import pyvista as pv
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from PIL import Image

# =============================================================================
# CONFIGURATION
# =============================================================================

OUTPUT_DIR = "Images_Output"
INPUT_DIR = "OUTPUT"

CLASS_NAMES = {
    0: "Unclassified",
    1: "Trunk",
    2: "Branch",
    3: "Twigs",
    4: "Grass"
}

CLASS_COLORS = {
    0: "#808080",  # Gray (unclassified)
    1: "#8B4513",  # Saddle Brown (trunk)
    2: "#CD853F",  # Peru/Orange (branch)
    3: "#4AC53F",  # Light Green (twigs)
    4: "#90EE90",  # Light Green (grass)
}

TREE_TITLES = {
    "01001_annotated": "Mattisleweiher site",
    "1_annotated": "Griese et al. (2025)",
    "19_annotated": "Griese et al. (2025)",
    "27_annotated": "Griese et al. (2025)"
}

# =============================================================================
# SETUP
# =============================================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)
las_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.las")))

if not las_files:
    print(f"No LAS files found in {INPUT_DIR}")
    exit()

print(f"Found {len(las_files)} LAS files")

# =============================================================================
# RENDER EACH LAS FILE
# =============================================================================

images = []
all_classes = set()
basenames = []

for las_file in las_files:
    print(f"Loading {las_file}...")
    las = laspy.read(las_file)
    
    # Extract coordinates
    points = np.vstack((las.x, las.y, las.z)).transpose()
    
    # Create PyVista PolyData
    cloud = pv.PolyData(points)
    
    # Get classification data
    class_data = np.array(las.classification).astype(np.int32)
    cloud.point_data["classification"] = class_data
    
    unique_classes = sorted(np.unique(class_data).tolist())
    all_classes.update(unique_classes)
    print(f"  {len(points)} points, classes: {unique_classes}")
    
    # Create colormap for classes present
    sorted_classes = sorted(unique_classes)
    colors = [CLASS_COLORS.get(c, "#808080") for c in sorted_classes]
    cmap = ListedColormap(colors)
    
    # Remap class values to contiguous indices for colormap
    class_mapping = {c: i for i, c in enumerate(sorted_classes)}
    remapped_data = np.array([class_mapping[c] for c in class_data], dtype=np.int32)
    cloud.point_data["classification"] = remapped_data
    
    # Create off-screen plotter
    plotter = pv.Plotter(off_screen=True)
    plotter.background_color = "white"
    
    plotter.add_mesh(
        cloud, 
        scalars="classification", 
        cmap=cmap, 
        categories=True, 
        show_scalar_bar=False,
        point_size=2.0,
        render_points_as_spheres=False,
        lighting=False,
        opacity=0.6
    )
    
    plotter.camera_position = 'iso'
    plotter.reset_camera()
    
    # Get image as numpy array
    img = plotter.screenshot(return_img=True)
    plotter.close()
    
    # Crop whitespace
    gray = np.mean(img[:, :, :3], axis=2)
    rows = np.any(gray < 250, axis=1)
    cols = np.any(gray < 250, axis=0)
    
    if np.any(rows) and np.any(cols):
        row_min, row_max = np.where(rows)[0][[0, -1]]
        col_min, col_max = np.where(cols)[0][[0, -1]]
        margin = 20
        row_min = max(0, row_min - margin)
        row_max = min(img.shape[0], row_max + margin)
        col_min = max(0, col_min - margin)
        col_max = min(img.shape[1], col_max + margin)
        img = img[row_min:row_max, col_min:col_max]
    
    images.append(img)
    basenames.append(os.path.splitext(os.path.basename(las_file))[0])

# =============================================================================
# CREATE COMPOSITE IMAGE
# =============================================================================

print("\nCreating composite image...")

# Find max height to align all images
max_height = max(img.shape[0] for img in images)

# Resize all images to same height
resized_images = []
for img in images:
    h, w = img.shape[:2]
    scale = max_height / h
    new_w = int(w * scale)
    pil_img = Image.fromarray(img)
    pil_img = pil_img.resize((new_w, max_height), Image.LANCZOS)
    resized_images.append(np.array(pil_img))

# Create composite with small gap
gap = 3
total_width = sum(img.shape[1] for img in resized_images) + gap * (len(resized_images) - 1)
composite = np.full((max_height, total_width, 3), 255, dtype=np.uint8)

x_offset = 0
for img in resized_images:
    w = img.shape[1]
    composite[:, x_offset:x_offset+w] = img[:, :, :3]
    x_offset += w + gap

# Create legend
legend_patches = []
for cls in sorted(all_classes):
    color = CLASS_COLORS.get(cls, "#808080")
    name = CLASS_NAMES.get(cls, f"Class {cls}")
    legend_patches.append(mpatches.Patch(color=color, label=name))

# Create final figure
fig_height = 6.5
fig_width = fig_height * (total_width / max_height)
fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))
ax.imshow(composite)
ax.axis('off')

# Add titles above each tree
x_offset = 0
for img, name in zip(resized_images, basenames):
    w = img.shape[1]
    center_x = x_offset + w / 2
    title = TREE_TITLES.get(name, name)
    ax.text(center_x, -20, title, ha='center', va='bottom', 
            fontsize=10, fontstyle='italic', color='#333333', fontweight='medium')
    x_offset += w + gap

# Add legend below
fig.legend(
    handles=legend_patches, 
    loc='lower center', 
    ncol=len(legend_patches),
    fontsize=10,
    frameon=True,
    fancybox=True,
    shadow=True,
    edgecolor='#cccccc',
    bbox_to_anchor=(0.5, 0.02)
)

plt.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.12)

# Save
output_path = os.path.join(OUTPUT_DIR, "all_trees_composite.png")
plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
plt.close()

print(f"\nSaved: {output_path}")
