# Modern Methods with TLS and UAV – Group 1

This repository contains the scripts used to train and apply a multi-channel YOLO model for tree component detection from terrestrial laser scanning (TLS) data.

The workflow converts 3D point clouds into 2D slices, trains a detection model on annotated slices, and projects predictions back into 3D point clouds.

A detailed explanation of the methods can be found in the project [report](https://georgerohan001.github.io/Modern-Methods-Group-2/motivation.html):

This README only explains **how to reproduce the workflow using the scripts provided**.

---

# 1. Clone the Repository

Clone the repository to your machine.

```bash
git clone https://github.com/georgerohan001/Modern-Methods-Group-2.git
cd Modern-Methods-Group-2
```

---

# 2. Obtain a Point Cloud

You may use the same dataset used in this project.

Download [example TLS data](https://bwsyncandshare.kit.edu/public.php/dav/files/79nqaSBT2Y6tkYH/2026-03-02_Mathisleweiher.laz).

Convert it to `.las` if necessary.

---

# 3. Split the Segmented Dataset into Individual Trees

Script: `Segmentor.R`

This script divides an already segmented TLS dataset into **one LAS file per tree**.  
The input point cloud must already contain a field with tree instance IDs (for example `preds_instance_segmentation`).  
The script reorganizes the dataset so that each tree can be loaded independently in CloudCompare.

## Install Dependencies

Open R and install the required package:

```r
install.packages("lidR")
````

Load it:

```r
library(lidR)
```

## Edit the Script

Modify the following variables:

```
input_file <- "3dtree_404_3596_segmentation.laz"
tree_field <- "preds_instance_segmentation"
```

* `input_file` → path to the segmented TLS point cloud
* `tree_field` → attribute containing the tree instance IDs

The script will automatically create three folders:

```
tiles
tree_parts
trees_split
```

The final individual trees will be written to `trees_split`.

## Run

Run the script in R:

```
source("Segmentor.R")
```

The script first divides the large point cloud into manageable tiles.
Within each tile, points are grouped by their tree ID and written as temporary tree parts.
These parts are then merged so that each tree is reconstructed as a single file.

The output folder will contain files such as:

```
tree_00001.las
tree_00002.las
tree_00003.las
```

Each file contains one individual tree and can be loaded independently.

---

# 4. Convert Trees to 2D Slices

Script: `slicer.py`

This script converts each tree LAS file into **horizontal 2D slices**.

Each slice is saved as a PNG image.

## Install Dependencies

```bash
pip install laspy numpy opencv-python
```

## Edit the Script

Modify the input folder:

```
INPUT_DIRECTORY = "path_to_folder_with_tree_las_files"
```

This should point to the directory created by `Segmentor.R`.

## Run

```
python slicer.py
```

## Output

```
tree_slices/
 ├─ tree_name_slice_000.png
 ├─ tree_name_slice_001.png
 ├─ tree_name_slice_002.png
 └─ metadata.json
```

The metadata file stores coordinate information used later for 3D reconstruction.

---

# 5. Annotate the Slices

Upload the generated PNG slices to **CVAT** for annotation.

Recommended export format:

```
YOLO for images 1.1
```

Create bounding boxes for the following classes:

```
0 → twigs
1 → trunk
2 → branch
3 → grass
```

---

# 6. Create Additional Image Channels

Script: `organize.py`

This script builds three additional image channels from the original slices.

These channels provide contextual information for the model.

## Install Dependencies

```bash
pip install pillow numpy
```

## Required Folder Structure

Create a base folder containing:

```
channel0
channel1
channel2
channel3
```

Place the **original slice PNG files** in:

```
channel0
```

## Edit the Script

Modify the base path:

```
BASE_DIR = "path_to_channel_directory"
```

## Run

```
python organize.py
```

## Output

The script will generate:

```
channel0 → original slices
channel1 → shifted slices
channel2 → shifted slices
channel3 → index-gradient channel
```

---

# 7. Combine Channels into Multi-Channel Images

Script: `combine_channels.py`

This script stacks the four grayscale channels into **4-channel TIFF images**.

## Install Dependencies

```bash
pip install pillow numpy tqdm
```

## Edit the Script

Modify:

```
BASE_DIR = "path_to_channel_directory"
```

## Run

```
python combine_channels.py
```

## Output

```
multichannel_tifs/
   tree_slice_000.tif
   tree_slice_001.tif
   tree_slice_002.tif
```

These TIFF files contain the four channels used for training.

---

# 8. Create a 4-Channel YOLO Checkpoint

Script: `make_4ch_checkpoint.py`

The default YOLO model accepts **3 channels (RGB)**.

This script modifies the pretrained weights so the model accepts **4 channels**.

## Install Dependencies

```bash
pip install torch
```

## Edit the Script

Update the two paths:

```
PRE = "path_to_original_yolo11s.pt"
OUT = "path_to_output_4ch_checkpoint.pt"
```

## Run

```
python make_4ch_checkpoint.py
```

The output file will be used during training.

---

# 9. Configure the Dataset

Files:

```
data.yaml
yolo11.yaml
```

## data.yaml

Update the dataset paths.

Example structure:

```
dataset/
 ├─ images/
 │   ├─ train
 │   └─ val
 └─ labels/
     ├─ train
     └─ val
```

Ensure the following parameters exist:

```
nc: 4
channels: 4
```

## yolo11.yaml

This model configuration defines the YOLO architecture.

Ensure:

```
channels: 4
```

---

# 10. Train the Model

Script: `train_4ch_detection.py`

This script trains the multi-channel YOLO model.

## Install Dependencies

```bash
pip install ultralytics torch torchvision
```

## Edit the Script

Update the paths:

```
CUSTOM_YAML = "path_to_yolo11.yaml"
DATA_YAML   = "path_to_data.yaml"
```

## Run

```
python train_4ch_detection.py
```

Training output will be stored in:

```
runs/detect/train/
```

The trained weights will be saved as:

```
best.pt
```

---

# 11. Run Predictions

Script: `predict_on_train.py`

This script runs the trained model on image slices.

## Edit the Script

Update:

```
MODEL_PATH
SOURCE_DIR
```

* `MODEL_PATH` → trained `best.pt`
* `SOURCE_DIR` → folder containing test images

## Run

```
python predict_on_train.py
```

The script will generate predicted bounding boxes and YOLO label files.

---

# 12. Project Predictions Back to 3D

Script: `aggregate.py`

This script converts the 2D detections back into **3D point cloud classifications**.

## Install Dependencies

```bash
pip install laspy numpy
```

## Edit the Script

Modify:

```
LAS_FILE
XML_FILE
METADATA_FILE
OUTPUT_FILE
```

These should correspond to:

* the original point cloud
* CVAT annotations
* slicer metadata
* output LAS file

## Run

```
python aggregate.py
```

The resulting LAS file contains classified points that can be visualized in CloudCompare.

---

# 13. Running the Complete Workflow Automatically

If you do not want to run each step separately, the repository also includes `run.py`.

This script combines most of the processing steps into a single pipeline.

## Folder Structure

Inside the `Workflow` folder:

```
Workflow
 ├─ INPUT
 ├─ OUTPUT
 ├─ Images
 ├─ Channels
 ├─ Labels
 ├─ COMPLETED
 └─ Model
```

Place your trained model in:

```
Workflow/Model/best.pt
```

Place LAS files to process in:

```
Workflow/INPUT
```

## Run

```
python run.py
```

The script will:

1. Slice the point cloud
2. Generate image channels
3. create multi-channel images
4. run YOLO predictions
5. project detections back into 3D

The final annotated LAS files will appear in:

```
Workflow/OUTPUT
```

# 14. Evaluate Model Performance

The repository includes scripts to evaluate both pixel-level and object-level metrics. These help analyze the performance of the trained YOLO model in predicting tree components.

---

## 14.1 Pixel-Level Metrics Evaluation

Script: `pixel_metrics.py`

This script calculates **pixel-level metrics**

### Install Dependencies

Ensure the required libraries are installed:
```bash
pip install laspy numpy pandas opencv-python matplotlib ultralytics pillow
```

### Input Requirements

1. **LAS Files**: Segmented point cloud files, placed in the `TEST` directory.
2. **Predictions**: YOLO prediction files (stored in `PREDICTED` during step 11).
3. **Metadata JSON**: Generated during slicing (step 4).

### Run the Script

Run the evaluation:
```bash
python pixel_metrics.py
```

### Outputs

1. **Confusion Matrices**: 
   - Visualization of prediction-vs-ground-truth classifications.
   - Stored for each tree in:
     ```
     OUTPUT/pixel_metrics/confusion_matrices/<tree_name>/
     ```
   - Example files:
     - `confusion_matrix_normalized.png`

2. **Overall Statistics**:
   - Summary metrics are **printed on the console**.
   - Metrics include precision, recall, F1-score, Pixel Accuracy, Foreground Accuracy, and mIoU.

3. **Per-Tree Details**:
   - Individual results for each tree, including class-level statistics, stored alongside confusion matrices.

---

## 14.2 YOLO Object-Level Metrics Evaluation

Please run `pixel_metrics.py` first!

Script: `yolo_metrics.py`

This script computes **object-level metrics** using YOLO’s built-in evaluation tools. It evaluates the detection accuracy of the model at the bounding box level.

### Metrics Generated

- **Detection Metrics**:
  - `mAP50`: Mean Average Precision at IoU threshold of 50%.
  - `mAP75`: Mean Average Precision at IoU threshold of 75%.
  - `mAP50-95`: Average Precision across IoU thresholds from 50% to 95% (COCO-style evaluation).
- **Box-Level Metrics**:
  - Mean Precision (MP)
  - Mean Recall (MR)
  - F1-score

### Install Dependencies

Install the required libraries:
```bash
pip install ultralytics torch pathlib
```

### Input Requirements

1. **Images and Labels**: Dataset structured with images and YOLO labels in `<DATASET_ROOT>/images` and `<DATASET_ROOT>/labels` directories.
2. **Model Weights**: Provide the trained YOLO model weights in `Model/best.pt`.

### Run the Script

```bash
python yolo_metrics.py
```

### Outputs

1. **Printed Metrics**:
   - Detailed metrics such as `mAP50`, `mAP50-95`, `mAP75`, mean precision, mean recall, and F1-score are printed on the console.

2. **Dataset Information**:
   - The prepared dataset structure, including links to images and labels, is stored in:
     ```
     OUTPUT/yolo_metrics/tmp_dataset_all
     ```
   - `data.yaml` file is generated within this directory, detailing the dataset configuration.

These metrics provide insights into how well the model is detecting tree components at both pixel-level and object-level resolutions.

---

If you need methodological details, refer to the [project report](https://georgerohan001.github.io/Modern-Methods-Group-2/motivation.html).
