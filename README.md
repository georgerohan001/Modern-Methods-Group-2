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

## 📂 6️⃣ Exporting from CVAT & Running the **Channel‑Creator** Pipeline  

From now on the only information we need from CVAT is the **YOLO‑1.1 annotation format**.  
All other image‑channel generation (index, label‑mask, trunk‑gradient) is handled by the
`channel_creator.py` script you just updated.

### Export the annotated slices from CVAT  

1. Open the finished task in **CVAT**.  
2. Click **`Menu → Export Task Dataset`**.  
3. In the **Export Format** drop‑down choose **`YOLO for images 1.1`**.  
4. Press **`Export`** → a **`.zip`** file will be downloaded.  

> **Why YOLO‑1.1?**  
> This format stores one `.txt` file per image, each line containing  
> `class_id cx cy w h` (all values normalised to the image size), which is exactly
> what our script expects.

### Unzip the export and check the folder layout  

```text
my_cvat_export/
│
├── obj.names                # class list (e.g. twigs, trunk, branch, grass)
├── train.txt                # list of image file names (optional – can be empty)
│
└── obj_train_data/
    ├── slice_000.png
    ├── slice_000.txt
    ├── slice_001.png
    ├── slice_001.txt
    └── … (one .png + one .txt per slice)
```

If the archive contains an extra top‑level folder, move its contents so that the
structure above is **directly inside** `my_cvat_export/`.

### Install the (only) required Python packages  

```powershell
pip install --upgrade pillow numpy
```

> The script uses **Pillow** for image I/O and **NumPy** for the index‑gray channel.

### Run the pipeline  

```powershell
cd path\to\my_cvat_export
python path\to\channel_creator.py
```

The script will:

| Action | Resulting folder | What you’ll find inside |
|--------|------------------|--------------------------|
| **Move** PNGs to `images/` and rename `obj_train_data/` → `annotations/` | `images/` & `annotations/` | Original slices (unchanged) and their YOLO `.txt` files. |
| **Create** `channel1/` (delete last image of each group, renumber) | `channel1/` | First set of processed PNGs. |
| **Create** `channel2/` (copy from `channel1/`, delete last again, renumber) | `channel2/` | Second set of processed PNGs. |
| **Create** index‑gray images (one per slice) | `channel3/` | 8‑bit grayscale images whose pixel value encodes the slice order. |
| **Validate / reorder** `obj.names` and fix all annotation files in `annotations/` | (in‑place) | Guarantees the class order `twigs → trunk → branch → grass`. |

When the script finishes you will have the three required folders:

```
my_cvat_export/
│
├─ images/            ← original PNGs (for reference)
├─ annotations/      ← repaired YOLO .txt files
├─ channel1/          ← first processed image set
├─ channel2/          ← second processed image set
└─ channel3/          ← index‑gray images (one per slice)
```

### What to commit (and what to ignore)  

* **Commit**  
  * `obj.names` (now guaranteed to be in the correct order)  
  * `annotations/` (the corrected YOLO label files)  
  * The **script** `channel_creator.py` (and any other helper scripts)  

* **Do **_not_** commit**  
  * All `*.png` files inside `images/`, `channel1/`, `channel2/`, `channel3/` – they can be regenerated any time.  
  * The original raw point‑cloud or any other large binary assets.  

Add the following lines to `.gitignore` (if they are not already present):

```gitignore
# CVAT export – large image folders
images/
channel1/
channel2/
channel3/
```

---


# 🤖 7️⃣ Moving Data to the Training Environment & Running YOLO

After generating the final folders (`annotations/`, `channel1/`, `channel2/`, `channel3/`) using `channel_creator.py`, the data must be moved to the **main training directory**.

You can find the training directory either:

- 🌐 **Google Drive (Shared Folder)**  
  https://drive.google.com/drive/folders/1w1j9_Zf_ZMcJTtCCYLx2jFmCjrDw_Qdr  

- 🖥️ **Workstation downstairs** (recommended for GPU training)

---

## 📁 1️⃣ Move the Processed Data

Copy your processed dataset into the main training directory.

Make sure:

- Image files are placed in the correct `images/` subfolders.
- Annotation `.txt` files are placed in the corresponding `labels/` subfolders.
- Filenames of images and labels match exactly.
- The train/validation split remains consistent.

---

## 📝 2️⃣ Verify the `data.yaml` File

Before training, open the `data.yaml` file in the training directory.

Check that:

- The dataset `path` is correct.
- The `train` and `val` paths point to the correct folders.
- The class order matches:

```

---

## 🔧 3️⃣ Install Required Python Packages

On the workstation (or your local GPU machine), install:

```powershell
pip install ultralytics torch torchvision
```

Optional but recommended:

```powershell
pip install matplotlib pandas seaborn
```

To verify GPU availability:

```powershell
python -c "import torch; print(torch.cuda.is_available())"
```

---

## ⚙️ 4️⃣ Adjust the Training Script

The file `train_yolo.py` is available in the training directory.

Before running it:

1. Open `train_yolo.py`
2. Locate the section where the working directory (`dataset_path`) is defined
3. Update it so it points to the correct dataset location on your machine

Example:

```python
dataset_path = r"C:\Path\To\datasets"
```

Save the file after editing.

---

## 🚀 5️⃣ Start Training

From the training directory, run:

```powershell
python train_yolo.py
```

For full training (300 epochs), use the **GPU workstation downstairs** whenever possible.
