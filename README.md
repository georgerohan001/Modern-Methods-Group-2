# Modern Methods with TLS and UAV Group 2

---

## 🚀 How to Contribute to This Project

To keep our forestry data and scripts organized, please follow this standard Git workflow every time you work on the project.

### 1. Initial Setup (First time only)

If you haven't downloaded the project to your computer yet, run:

```powershell
git clone https://github.com/georgerohan001/Modern-Methods-Group-2.git

```

### 2. The Daily Workflow

Follow these steps **every time** you want to make a change:

#### **Step A: Sync with the Group**

Before you start typing any code or moving any TLS files, make sure you have the latest version from everyone else.

```powershell
git pull origin main

```

#### **Step B: Make Your Changes**

Open your files, run your processing scripts, or add your UAV data analysis. Save your files as usual.

#### **Step C: Stage the Changes**

Tell Git which files you want to prepare for the upload. The `.` means "add everything."

```powershell
git add .

```

#### **Step D: Commit the Work**

Create a "snapshot" of your work with a clear message describing what you did (e.g., "Added noise filter to LiDAR script").

```powershell
git commit -m "Brief description of what you changed"

```

#### **Step E: Push to the Cloud**

Upload your snapshot so the rest of the group can see it.

```powershell
git push origin main

```

---

### ⚠️ Important Rules for This Project

* **Pull Before Push:** Always run `git pull` before you start working to avoid merge conflicts.
* **Small Commits:** It is better to push 5 small changes than one giant update. It makes it easier to fix if something breaks!
* **Large Data:** Do **not** commit raw `.las` or `.tif` files larger than 50MB. Use the shared cloud drive for raw data and Git for the processing scripts.

---

## 🌳 Point Cloud Slicing Tool

This repository includes a Python script to convert `.las` point clouds into a stack of 2D density PNG slices. This is useful for analyzing tree structure, DBH (Diameter at Breast Height), or crown volume.

### 1. Prerequisites

Ensure you have the required Python libraries installed:

```powershell
pip install laspy numpy opencv-python

```

### 2. How to Use the Slicer

1. **Prepare the Data:** Copy the Python script into the same folder as your `.las` file (e.g., `group_321_GP.las`).
2. **Edit the Script:** Open the script and update the `FILE_NAME` variable to match your file:
```python
# --- CONFIGURATION ---
FILE_NAME = "your_actual_filename_here.las"

```


3. **Run the Script:**
```powershell
python your_script_name.py

```



### 3. What Happens Next?

* **Automatic Slicing:** The script calculates the "global bounds" of your tree so every image is the exact same width and height.
* **Output:** A new folder named `tree_slices/[tree_name]` will be created.
* **Files:**
* **PNGs:** Each image represents a vertical slice (default is 20cm thick).
* **Metadata.json:** Contains the spatial coordinates (X, Y, Z) and pixel size for every slice so you can map pixels back to real-world coordinates.



---

### ⚠️ A Note for the Team (Git Large Files)

**Important:** Do **not** commit the generated `tree_slices/` folder or large `.las` files to GitHub. These files are too large for Git and will slow down everyone's `pull` and `push` times.

To keep our repo clean, ensure your `.gitignore` includes:

```text
*.las
tree_slices/

```

---

## 🏗️ Re-aggregating Annotations (3D Reconstruction)

Once you have finished labeling your 2D slices in **CVAT**, use the `tree_aggregate.R` script to project those labels back onto the original 3D point cloud. This allows you to visualize your "trunk," "branch," and "twig" classifications in 3D.

### 1. Prerequisites

You will need **R** installed. Open your R terminal or RStudio and install the necessary spatial and data libraries:

```r
install.packages(c("lidR", "xml2", "jsonlite", "data.table"))

```

### 2. Exporting from CVAT

1. Open your project in CVAT.
2. Go to **Menu > Export Task Dataset**.
3. Select the **CVAT for images 1.1** (XML) format.
4. Download and unzip the file into your project directory.

### 3. Running the Aggregator

Before running the script, open `tree_aggregate.R` and update the **USER SETTINGS** section (Lines 11-16) to match your local file names:

```r
# --- CHANGE THESE TO MATCH YOUR PROJECT ---
las_file      <- "your_original_file.las"       # The raw point cloud
xml_file      <- "path/to/annotations.xml"      # The file you got from CVAT
metadata_file <- "path/to/metadata.json"        # The JSON from the Python slicer
output_file   <- "annotated_tree.las"           # The name of your new 3D file

```

Run the script in your PowerShell or R terminal:

```powershell
Rscript tree_aggregate.R

```

### 4. Viewing the Results

The script creates a new `.las` file where the **Classification** field of each point is updated based on your CVAT labels:

* **1**: Trunk
* **2**: Branch
* **3**: Twigs
* **4**: Grass

**💡 Pro-Tip:** Open the resulting file in **CloudCompare**. To see your work, change the "Color Scale" or "Scalar Field" view to **Classification**.

