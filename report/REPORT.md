# Branch and Trunk Detection in Single-Tree Point Clouds using YOLOv11

- **Course:** Modern Methods with TLS and UAV
- **Group:** Group 2
- **Date:** 2026-03-11

---

## 1. Introduction / Motivation

- For many forestry applications, decomposing a single-tree point cloud into structural components (branches, trunk, twigs, grass/ground) is necessary
- Applications: trunk volume estimation, taper functions, wood quality metrics (e.g. bending), biomass modelling
- Approach: transform the 3D detection problem into 2D object detection by slicing the point cloud horizontally and detecting cross-sectional shapes on 2D density rasters
- Object detection chosen over segmentation (see Section 7.1 for discussion)

---

## 2. Data

### 2.1 Point Cloud Data

- Source: `.las` files of individual trees (already segmented from forest point cloud)
- Sensor type: [TODO: specify sensor type, e.g. TLS/MLS/ALS]
- Location / forest type: [TODO: location, species, stand type]
- Number of trees: [TODO: N trees total]
- Reference files visible in code: `group_321_GP.las`, `tree_00844.las`, `tree_00638`
- Dataset name in pipeline: `my_yolo_dataset` (571 images noted in README)

### 2.2 Point Cloud Slicing (`Slicer.py`)

- Slice thickness: **0.20 m** (fixed, `SLICE_HEIGHT = 0.20`)
- Pixel size: **0.01 m/pixel** (1 cm resolution, `PIXEL_SIZE = 0.01`)
- Method: 2D point density histogram (`np.histogram2d`) projected on XY plane
- Normalisation: 99th percentile clipping → scaled to 0–255 uint8
- Canvas: global bounds per tree → all slices same spatial extent and pixel dimensions
- Output: PNG per slice + `metadata.json` (per-slice: `x_origin_global`, `y_origin_global`, `z_layer`, `pixel_size`, `canvas_w`, `canvas_h`)
- ~150 slices per tree (depending on tree height)
- Total slices: [TODO: total number across all trees]

### 2.3 Annotation (CVAT)

- Tool: CVAT (Computer Vision Annotation Tool)
- Export format for backtranslation: **CVAT for images 1.1** (XML)
- Export format for YOLO training: **YOLO for images 1.1** (`.txt` files)
- Classes (4 total, order enforced by `organize.py`: `twigs=0, trunk=1, branch=2, grass=3`):
  - `trunk` (class 1): main stem cross-section
  - `branch` (class 2): lateral branches
  - `twigs` (class 0): small/thin branches
  - `grass` (class 3): ground vegetation
- Annotation type: axis-aligned **bounding boxes** per visible cross-section
- Total bounding boxes: [TODO: N boxes total]
- Annotators: [TODO: names or "group members"]
- QC / review process: [TODO: describe or "visual check by all group members"]

### 2.4 Train / Validation / Test Split

- Split rule: **tree-level** — no slices from the same tree appear in multiple splits (prevents data leakage)
- Split ratio: [TODO: e.g. 70/15/15 or 80/10/10]
- N trees per split: [TODO: train/val/test tree counts]
- N slices per split: [TODO: train/val/test slice counts]
- Test set: **no labels** → visual evaluation only (time constraints)
- Validation set: labelled → used for mAP metrics during training

---

## 3. Preprocessing Pipeline

### 3.1 Multi-Channel Feature Engineering

Three additional channels were engineered beyond the raw density slice to provide contextual and spatial information to the model:

| Channel | Name | Content | Script |
|---------|------|---------|--------|
| Ch 0 | Primary slice | Point density raster of slice `i` | `Slicer.py` |
| Ch 1 | Next slice (i+1) | Density raster of the slice directly above | `combine_channels.py` |
| Ch 2 | Slice +2 (i+2) | Density raster two levels above | `combine_channels.py` |
| Ch 3 | Index-gray | Flat image where all pixels = `slice_index` (0–255, linear mapping over full stack) | `generate_index_gradient_images.py`, `organize.py` |

- Rationale for Ch1/Ch2: provide vertical context — branches taper upward, making adjacent slices informative
- Rationale for index-gray (Ch3): encodes absolute height position of the slice within the tree; helps distinguish trunk base (bottom) from crown (top)
- Format: 4-channel TIFF (`RGBA` mode, `tiff_deflate` compression), shape `(H, W, 4)`
- Implementation: `combine_channels.py`
- Missing neighbours (top of tree): filled with zeros (black channel)

### 3.2 Dataset Organisation (`organize.py`)

- Moves PNGs from `obj_train_data/` → `images/`; renames `obj_train_data/` → `annotations/`
- Creates `channel1/`, `channel2/`, `channel3/`
- `channel1`: copy of `images/`, last slice of each tree deleted, renumbered
- `channel2`: copy of `channel1/`, last slice deleted again, renumbered
- `channel3`: index-gray images (slice position encoded as constant grayscale value 0–255)
- Enforces class name order: `["twigs", "trunk", "branch", "grass"]` in `obj.names`; rewrites all annotation `.txt` files to match

### 3.3 Additional Channel Helper (`channel_creator.py`)

Generates three additional feature images from CVAT YOLO exports:

- **Index-gray**: linear grayscale encoding of slice position
- **Label mask**: filled bounding boxes rendered as grayscale levels per class (twigs=255, trunk=205, branch=155, grass=105)
- **Gradient image**: distance-to-trunk gradient — brightness falls off with distance from trunk centre
- Gradient: piecewise function: near trunk (r≤0.25): 255→127.5; far (r>0.25): 127.5→0

---

## 4. Model

### 4.1 Architecture: YOLO11s (3-channel, standard)

- Framework: Ultralytics YOLO11 (`ultralytics` library)
- Model variant: **YOLO11s** ("small")
- Justification for `s` variant:
  - Fast training on available hardware
  - Sufficient capacity for relatively simple 2D cross-section patterns
  - [TODO: compare with larger variants in future work]
- Input channels: **3** (standard RGB, used for initial training run)
- Input resolution: **640×640** px
- Number of classes: **4** (`twigs`, `trunk`, `branch`, `grass`)
- Pretrained weights: `yolo11s.pt` (COCO pretrained, fine-tuned)

### 4.2 Architecture: YOLO11s-4ch (4-channel, modified)

- Same base as YOLO11s but first Conv layer replaced: `in_channels` changed from 3 → 4
- Weight initialisation for new 4th channel: **mean of original RGB weights** (`first_conv.weight.mean(dim=1, keepdim=True)`) → avoids random initialisation
- Original RGB weights fully transferred (no weight reset)
- Created via `create_4channel_model.py`
- Saved as `yolo11s_4ch.pt`
- Input: 4-channel TIFFs (Ch0–Ch3 as described above)
- Status: **[TODO: training completed / in progress / pending]**

---

## 5. Training

### 5.1 Standard 3-channel run (`train_yolo.py`)

| Parameter | Value |
|-----------|-------|
| Epochs configured | 300 |
| Epochs run | ~189 (training stopped early — low improvement) |
| Early stopping patience | 30 epochs |
| Batch size | 32 |
| Image size | 640 px |
| Initial LR (`lr0`) | 0.01 |
| LR schedule | cosine decay (Ultralytics default) |
| Augmentation | enabled (`augment=True`) |
| Rectangular batches | enabled (`rect=True`) |
| Cache | enabled |
| GPU | [TODO: GPU model, e.g. NVIDIA RTX XXXX] |
| Training time | ~4089 s total (epoch 189 at 4089 s cumulative) |
| Best model | `best.pt` saved automatically |

**Training metrics at final epoch (189):**

| Metric | Value |
|--------|-------|
| train/box_loss | 1.149 |
| train/cls_loss | 0.729 |
| train/dfl_loss | 1.043 |
| val/box_loss | 1.418 |
| val/cls_loss | 1.231 |
| val/dfl_loss | 1.178 |
| metrics/precision(B) | 0.788 |
| metrics/recall(B) | 0.471 |
| metrics/mAP50(B) | 0.543 |
| metrics/mAP50-95(B) | 0.335 |
| LR (all param groups) | 0.000474 |

**Training curve observations:**

- Epochs 1–5: highly unstable — `val/cls_loss` shows `inf`/`nan` values (empty slices, very few foreground objects); model collapses briefly (precision/recall → 0)
- Epoch 13 onwards: stable training — val losses converge, metrics improve consistently
- Losses (train & val) decrease monotonically from epoch ~13 to ~189
- Precision reaches ~0.78 by epoch 189; recall lags behind at ~0.47 (model is conservative — misses detections but avoids false positives)
- mAP50 plateau around 0.50–0.57 in later epochs; mAP50-95 plateau around 0.31–0.34
- Early stopping triggered around epoch 189 (patience=30, last significant improvement ~[TODO: epoch of best checkpoint])
- LR decays smoothly from 0.00123 (peak) to 0.000474 following cosine schedule

### 5.2 4-channel run (`train_multichannel.py`)

| Parameter | Value |
|-----------|-------|
| Model | `yolo11s_4ch.pt` |
| Epochs configured | 10 (test run) |
| Patience | 5 |
| Batch size | 32 |
| Image size | 640 px |
| Dataset | `my_yolo_dataset` (571 images) |
| Status | [TODO: results pending / completed] |
| Best mAP50 | [TODO] |
| Best mAP50-95 | [TODO] |

---

## 6. Results

### 6.1 Quantitative Results (Validation Set — 3-channel model)

| Metric | Best epoch value |
|--------|-----------------|
| mAP50(B) | ~0.653 (epoch 140) |
| mAP50-95(B) | ~0.427 (epoch 140) |
| Precision | ~0.775 |
| Recall | ~0.467 |

- Best epoch by mAP50: epoch 140 (mAP50=0.653, mAP50-95=0.427)
- Note: validation metrics computed on labelled val split only; test set has no labels

### 6.2 Quantitative Results (4-channel model)

- [TODO: add results after training completes]
- Expected advantage: additional vertical context (Ch1, Ch2) and height encoding (Ch3) may improve recall and reduce false positives at crown

### 6.3 Qualitative / Visual Evaluation (Test Set)

- [TODO: add representative images of model predictions on test slices]
- [TODO: describe visual quality — e.g. trunk detection reliable, small branches missed, false positives near dense branch junctions]
- Evaluation method: visual inspection of YOLO inference output overlaid on PNG slices

### 6.4 3D Backtranslation Results

- [TODO: add visual results of annotated point cloud in CloudCompare]
- Method: CVAT XML → `tree_aggregrate.R` → classified `.las` file
- Classification codes: `1=trunk, 2=branch, 3=twigs, 4=grass`
- View in CloudCompare: Scalar Field → Classification

---

## 7. Discussion

### 7.1 Detection vs. Segmentation

- **Chosen approach**: bounding box detection (YOLO)
- **Key problem with segmentation** in this domain:
  - Branches overlap heavily in cross-section, especially in the crown
  - Overlapping segmentation masks create ambiguous labels — a single pixel can belong to multiple branches
  - Detection boxes can overlap freely without label conflict
- **Potential advantages of segmentation**:
  - More precise boundary delineation → better 3D backtranslation (tighter pixel-to-point mapping)
  - Could directly output branch area/diameter from mask rather than bounding box approximation
  - Instance segmentation (e.g. YOLO11-seg) would still handle overlapping instances
- **Conclusion**: detection was a pragmatic and valid starting choice; segmentation (instance-level) should be explored in future work

### 7.2 Model Size (YOLO11s vs. larger)

- `s` variant chosen for speed; `m` or `l` variants would have more capacity for distinguishing fine-grained classes (twigs vs. branch)
- [TODO: if time permits, compare `yolo11s` vs `yolo11m` on same dataset]

### 7.3 Multi-Channel Approach

- Novel approach: standard YOLO trained on images; adding neighbouring slice context is non-standard
- Ch3 (index-gray) provides height encoding implicitly — helps model learn that trunk appears at all heights, while grass only at base
- Limitation: first conv layer weights for Ch3 initialised as mean of RGB — not ideal; future work could use separate encoder for height channel

### 7.4 Training Instability (Early Epochs)

- `nan`/`inf` val losses at epochs 3, 5, 8, 9, 10: caused by slices with no foreground objects or validation batches where no GT boxes exist
- Workaround: model recovers by epoch 13; early stopping patience prevents premature termination
- Could be mitigated by filtering out empty slices from val set

### 7.5 Precision vs. Recall Trade-off

- Model shows higher precision (~0.78) than recall (~0.47) at final epoch
- Indicates conservative detection — misses detections more often than producing false positives
- Possible cause: class imbalance (many slices have only trunk, few have many branches)
- [TODO: analyse per-class AP to confirm]

---

## 8. Methods Summary (Pipeline Overview)

```
.las (single tree)
    └─ Slicer.py ──────────────────────────────────────────────────────┐
         slice_height=0.20m, pixel_size=0.01m                          │
         output: PNGs + metadata.json                                  │
                                                                        ▼
                                                               CVAT annotation
                                                               (bounding boxes)
                                                               4 classes: twigs,
                                                               trunk, branch, grass
                                                                        │
         ┌──────────────────────────────────────────────────────────────┘
         │
         ├─ organize.py / channel_creator.py / combine_channels.py
         │    → 4-channel TIFFs [Ch0: slice_i, Ch1: slice_i+1,
         │                        Ch2: slice_i+2, Ch3: index-gray]
         │
         ├─ train_yolo.py (3-ch YOLO11s)
         │    epochs=300, batch=32, imgsz=640, lr0=0.01
         │    → stopped at ~189 epochs, best mAP50≈0.653
         │
         ├─ create_4channel_model.py → yolo11s_4ch.pt
         │    (RGB weights transferred, 4th ch = mean of RGB)
         │
         └─ train_multichannel.py (4-ch YOLO11s)
              epochs=10 (test run), batch=32, imgsz=640
              → [TODO: results]

Inference on test slices
    → visual evaluation only (no GT labels on test set)

tree_aggregrate.R (backtranslation)
    CVAT XML + metadata.json → annotated .las
    → CloudCompare visualisation (scalar field: Classification)
```

---

## 9. Repository Structure

```
Modern-Methods-Group-2/
├── Slicer.py                          # Point cloud → 2D PNG slices + metadata.json
├── channel_creator.py                 # Generate index-gray, label mask, gradient images from CVAT export
├── generate_index_gradient_images.py  # Standalone: index-gray image generation
├── organize.py                        # Dataset folder restructuring + class index remapping
├── combine_channels.py                # Create 4-channel TIFFs from sequential slices
├── create_4channel_model.py           # Modify YOLO11s to accept 4 input channels
├── train_yolo.py                      # Train standard 3-channel YOLO11s
├── train_multichannel.py              # Train 4-channel YOLO11s
├── tree_aggregrate.R                  # 2D CVAT annotations → 3D point cloud labels (backtranslation)
├── requirements.txt                   # numpy, Pillow, laspy, opencv-python
├── results.csv                        # Training metrics per epoch (3-channel run, 189 epochs)
└── report/
    └── REPORT.md                      # This report
```

---

## 10. References / Dependencies

**Python packages:**
- `ultralytics` (YOLO11)
- `torch`, `torchvision`
- `laspy` (LAS point cloud I/O)
- `numpy`, `Pillow`, `opencv-python`
- `tqdm`

**R packages:**
- `lidR` (point cloud I/O and processing)
- `xml2` (CVAT XML parsing)
- `jsonlite` (metadata.json parsing)
- `data.table` (fast indexing)

**Tools:**
- CVAT (annotation)
- CloudCompare (3D visualisation)

**Literature / further reading:** [TODO: add relevant citations, e.g. YOLO paper, TLS tree segmentation papers]
