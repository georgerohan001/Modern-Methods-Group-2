# Modern Methods with TLS and UAV – Group 1

## Git Workflow

### Initial Setup
```powershell
git clone https://github.com/georgerohan001/Modern-Methods-Group-2.git
```

### Daily Workflow
```powershell
git pull origin main          # Sync before starting
# Make your changes
git add .                     # Stage all changes
git commit -m "Description"  # Commit with message
git push origin main          # Upload to remote
```

### Rules
- **Pull before push** – prevents merge conflicts
- **Small commits** – easier debugging and collaboration
- **No large files** – keep `.las`, `.tif` >50MB in shared cloud, not Git

---

## Point Cloud Slicing Tool

Converts `.las` point clouds into 2D PNG slices for tree structure analysis.

### Setup
```powershell
pip install laspy numpy opencv-python
```

### Usage
1. Place script with `.las` file
2. Update `FILE_NAME` in script
3. Run: `python slicer.py`

### Output
- `tree_slices/[tree_name]/` – PNG slices (20cm thickness default)
- `metadata.json` – X, Y, Z coordinates and pixel resolution for 3D reconstruction

---

## 3D Reconstruction (CVAT → Point Cloud)

Project CVAT annotations back onto original 3D point cloud.

### Prerequisites
```r
install.packages(c("lidR", "xml2", "jsonlite", "data.table"))
```

### Export from CVAT
`Menu → Export Task Dataset → CVAT for images 1.1`

### Run Aggregator
```r
# Edit tree_aggregate.R:
las_file      <- "your_original_file.las"
xml_file      <- "path/to/annotations.xml"
metadata_file <- "path/to/metadata.json"
output_file   <- "annotated_tree.las"
```
```powershell
Rscript tree_aggregate.R
```

### Classification Mapping
```
1 → Trunk | 2 → Branch | 3 → Twigs | 4 → Grass
```

View in CloudCompare: `Scalar Field → Classification`

---

## Channel-Creator Pipeline

Generates multi-channel training data from CVAT exports.

### Export from CVAT
`Menu → Export Task Dataset → YOLO for images 1.1`

### Install
```powershell
pip install --upgrade pillow numpy
```

### Run
```powershell
cd path\to\cvat_export
python path\to\channel_creator.py
```

### Output Structure
```
cvat_export/
├── images/        # Original PNGs
├── annotations/   # YOLO .txt files (class order: twigs, trunk, branch, grass)
├── channel1/      # Processed set 1
├── channel2/      # Processed set 2
└── channel3/      # Index-gray images
```

### Gitignore
```gitignore
images/
channel1/
channel2/
channel3/
```

## Updated: Multi-Channel TIFF Creation

Creates 4-channel TIFF files for YOLO11 multi-channel training.
### Requirements
```powershell
pip install numpy pillow tqdm
```

Usage
1. Ensure images are in data/datasets/datasets/tree_0638/images/train/ or you wanted location
2. Run:
python combine_channels.py
Output
- Location: data/datasets/datasets/tree_0638/multichannel/ -> change for different dataset
- Format: 4-channel TIFF (channels, height, width)
- Channels:
  - Ch 0: Primary slice
  - Ch 1: Slice i+1
  - Ch 2: Slice i+2
  - Ch 3: Index-gray (slice number as grayscale)
Training with Multi-Channel
Update data.yaml:
path: tree_0638
train: multichannel
val: multichannel
nc: 4
names: [twigs, trunk, branch, grass]
channels: 4
Then train:
python train_yolo.py

---

## YOLO Training

### Move Data
Copy processed dataset to training directory:
- Google Drive: https://drive.google.com/drive/folders/1w1j9_Zf_ZMcJTtCCYLx2jFmCjrDw_Qdr
- Workstation downstairs (recommended for GPU)

### Install
```powershell
pip install ultralytics torch torchvision
```

### Verify GPU
```powershell
python -c "import torch; print(torch.cuda.is_available())"
```

### Train
1. Update `dataset_path` in `train_yolo.py`
2. Run: `python train_yolo.py`

Use GPU workstation for full training (300 epochs).
