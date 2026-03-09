# Modern Methods with TLS and UAV – Group 1

---

# 🚀 How to Contribute to This Project

To keep our forestry data, annotations, and scripts organized, please follow this **standard Git workflow** every time you work on the project.

---

## 1. Initial Setup (First Time Only)

If you haven't downloaded the project to your computer yet, run:

```powershell
git clone https://github.com/georgerohan001/Modern-Methods-Group-2.git
```

This downloads the repository to your local machine.

---

## 2. The Daily Workflow

Follow these steps **every time you want to make a change**.

### Step A — Sync with the Group

Before writing code or processing any TLS/UAV data, always download the newest changes from the group.

```powershell
git pull origin main
```

This prevents merge conflicts and ensures everyone is working with the latest files.

---

### Step B — Make Your Changes

Edit scripts, run processing workflows, or add analysis outputs.

Save your files normally.

---

### Step C — Stage the Changes

Tell Git which files should be included in your update.

```powershell
git add .
```

The `.` means **add all changed files**.

---

### Step D — Commit the Work

Create a snapshot of your changes with a short explanation.

```powershell
git commit -m "Brief description of what you changed"
```

Example:

```
git commit -m "Added trunk detection gradient generator"
```

---

### Step E — Push to the Cloud

Upload your snapshot so the rest of the group can access it.

```powershell
git push origin main
```

---

## ⚠️ Important Rules for This Project

### Pull Before Push

Always run:

```powershell
git pull
```

before starting work.

---

### Small Commits

It is better to push **multiple small commits** instead of one large update.

Benefits:

* Easier debugging
* Easier collaboration
* Clear project history

---

### Large Data Files

Do **not commit large raw data files**.

Avoid uploading:

```
.las
.tif
```

files larger than **50 MB**.

Instead:

* Store raw data in the **shared cloud drive**
* Store **scripts and processing pipelines in Git**

---

# 🌳 Point Cloud Slicing Tool

This repository includes a Python script that converts `.las` point clouds into a stack of **2D density PNG slices**.

These slices are used for:

* Tree structure analysis
* DBH (Diameter at Breast Height)
* Crown volume estimation
* Image annotation workflows

---

## 1. Prerequisites

Install the required Python libraries:

```powershell
pip install laspy numpy opencv-python
```

---

## 2. How to Use the Slicer

### Step 1 — Prepare the Data

Place the slicer script in the same folder as your `.las` file.

Example:

```
group_321_GP.las
pointcloud_slicer.py
```

---

### Step 2 — Edit the Script

Open the script and update the configuration section:

```python
# --- CONFIGURATION ---
FILE_NAME = "your_actual_filename_here.las"
```

---

### Step 3 — Run the Script

```powershell
python your_script_name.py
```

---

## 3. What Happens Next?

The slicer performs several automatic steps.

### Automatic Tree Bounds

The script calculates the **global spatial bounds** of the tree.

This ensures **all slices have identical dimensions**.

---

### Output Folder

A folder is generated:

```
tree_slices/[tree_name]/
```

---

### Generated Files

Inside this folder you will find:

#### PNG Slices

Each PNG represents a **vertical slice of the tree**.

Default slice thickness:

```
20 cm
```

---

#### Metadata File

```
metadata.json
```

This file stores:

* X coordinate
* Y coordinate
* Z slice height
* pixel resolution

This allows **mapping pixels back to real-world coordinates**.

---

## ⚠️ Note for the Team (Git Large Files)

Do **not commit** the following to GitHub:

```
tree_slices/
*.las
```

These files are too large and will slow down the repository.

Make sure `.gitignore` contains:

```
*.las
tree_slices/
```

---

# 🏗️ Re-aggregating Annotations (3D Reconstruction)

After labeling the 2D slices in **CVAT**, you can project those labels back onto the **original 3D point cloud**.

This step reconstructs the labeled tree in **3D space**.

---

## 1. Prerequisites

Install required R packages:

```r
install.packages(c("lidR", "xml2", "jsonlite", "data.table"))
```

---

## 2. Exporting from CVAT

1. Open the task in **CVAT**
2. Click:

```
Menu → Export Task Dataset
```

3. Select format:

```
CVAT for images 1.1
```

4. Download and unzip the export.

---

## 3. Running the Aggregator

Open `tree_aggregate.R` and update the user configuration:

```r
# --- CHANGE THESE TO MATCH YOUR PROJECT ---
las_file      <- "your_original_file.las"
xml_file      <- "path/to/annotations.xml"
metadata_file <- "path/to/metadata.json"
output_file   <- "annotated_tree.las"
```

Run the script:

```powershell
Rscript tree_aggregate.R
```

---

## 4. Viewing the Results

The script produces a new `.las` file where each point receives a **classification value**.

Classification mapping:

```
1  → Trunk
2  → Branch
3  → Twigs
4  → Grass
```

---

### Visualizing in CloudCompare

Open the new `.las` file in **CloudCompare**.

Set the color mode to:

```
Scalar Field → Classification
```

This will display the tree with colored structural labels.

---

# 🎯 Channel Creator (Multi-Channel Training Image Generator)

The script **`channel_creator.py`** generates multiple grayscale training channels from the labeled slice images.

It replaces several older scripts by producing **all training channels in a single pipeline**.

The script reads **YOLO bounding box annotations** exported from CVAT and creates three derived images for each slice:

1. **Index Channel** – encodes slice order
2. **Label Mask Channel** – encodes object classes
3. **Trunk Gradient Channel** – distance heatmap from trunks

These channels are useful for **machine learning models that benefit from additional spatial context**.

---

## 1. Prerequisites

Install the required libraries:

```powershell
pip install pillow numpy
```

---

## 2. Required Dataset Structure

Your dataset must follow this structure:

```
YOLO_Extraction_1/
│
├── obj.names
├── obj.data
├── train.txt
│
├── obj_train_data/
│   ├── slice_000.png
│   ├── slice_000.txt
│   ├── slice_001.png
│   └── slice_001.txt
│
└── channel_creator.py
```

### Explanation

| File              | Purpose                          |
| ----------------- | -------------------------------- |
| `obj.names`       | Ordered list of class labels     |
| `train.txt`       | List of training image paths     |
| `obj_train_data/` | Images and YOLO annotation files |
| `.txt` files      | Bounding boxes in YOLO format    |

Example YOLO label:

```
1 0.77 0.27 0.04 0.05
```

Format:

```
[class_id center_x center_y width height]
```

Coordinates are **normalized (0–1)** relative to the image size.

---

## 3. Output Channels

Running the script creates **three output folders**.

```
YOLO_Extraction_1/
│
├── index_gray_images/
├── label_polygon_images/
└── gradient_images/
```

Each folder contains an image corresponding to every input slice.

---

### Channel 1 — Index Channel

Folder:

```
index_gray_images/
```

Each slice receives a grayscale value based on its **position in the dataset**.

```
first slice   → black
last slice    → white
```

This gives neural networks **vertical positional information** about the tree.

---

### Channel 2 — Label Mask Channel

Folder:

```
label_polygon_images/
```

This channel converts bounding boxes into **grayscale masks representing object classes**.

Example intensity mapping:

```
Trunk   → bright
Branch  → medium gray
Twig    → darker
Grass   → darker
```

Each object appears as a **filled rectangle** corresponding to its YOLO bounding box.

---

### Channel 3 — Trunk Gradient Channel

Folder:

```
gradient_images/
```

This channel generates a **distance-based heatmap** from trunk locations.

```
White  → trunk center
Gray   → near trunk
Black  → far from trunk
```

The gradient fades based on the distance to the **nearest trunk bounding box**.

This is useful for:

* trunk localization
* spatial loss weighting
* machine learning supervision signals

---

## 4. Running the Script

Open `channel_creator.py` and set the base path:

```python
BASE_DIR = Path(r"C:\Users\...\YOLO_Extraction_1")
```

Then run:

```powershell
python channel_creator.py
```

---

## 5. Processing Pipeline

For each image slice the script:

1. Reads the original slice image
2. Loads the YOLO label file
3. Extracts trunk bounding boxes
4. Generates the three training channels
5. Saves them to their output folders

All channels maintain the **exact same resolution as the original slice**.

---

## 6. Why Multi-Channel Training Helps

Providing additional channels can improve model learning because it adds structured spatial information:

| Channel    | Information            |
| ---------- | ---------------------- |
| Index      | Vertical tree position |
| Label mask | Class location         |
| Gradient   | Distance to trunk      |

This creates a richer representation than using raw slices alone.

---

## 7. Important Notes

Do **not commit generated image channels** if they become large.

If necessary, add them to `.gitignore`:

```
index_gray_images/
label_polygon_images/
gradient_images/
```

Generated outputs can always be **recreated using `channel_creator.py`**.