# 2. Data Acquisition & Preparation

This section details the transition from high-density 3D point clouds to the 2D image datasets required for YOLOv11 training.

---

## 2.1 Raw Point Cloud Data
The primary data source consists of individual tree point clouds extracted from larger forest scans.

* **Source Format:** `.las` / `.laz` files (pre-segmented individual trees).
* **Sensor Type:** RIEGL VZ-400i Terrestrial Laser Scanner (TLS).
* **Location:** Matthisleweiher, Black Forest, Germany.
* **Forest Type:** Mixed coniferous stand (predominantly *Picea abies*).
* **Dataset Size:** [N] total trees processed.

---

## 2.2 Point Cloud Slicing (The 2D Transformation)
To bridge the gap between 3D geometry and 2D computer vision, the point clouds underwent a rigorous stratified slicing and rasterization process.

### The Slicing Workflow
1.  **Vertical Stratification:** Each tree point cloud was discretized into **20 cm horizontal slices** along the Z-axis.
2.  **Projection:** Points within each slice were projected onto the **XY plane** to create a 2D density histogram.
3.  **Rasterization:** The density maps were resampled to a spatial resolution of **1 cm per pixel**.
4.  **Normalization:** To ensure consistent input for the neural network:
    * **99th Percentile Clipping:** Used to mitigate the influence of extreme point density outliers.
    * **Scaling:** Pixel values were scaled to **0–255 (uint8)**.
    * **Global Bounds:** All slices for a specific tree share the same spatial extent and pixel dimensions to maintain structural context.

> [!NOTE] Dataset Volume
> On average, this resulted in **~150 slices per tree**, depending on total tree height, resulting in a total dataset of **[TODO: Total Number]** images.

---

## 2.3 Annotation Protocol
Data labeling was performed using **CVAT** (Computer Vision Annotation Tool), with labels exported in the **YOLO 1.1** format.

### Target Classes
We defined four distinct classes for the object detection task:
1.  **Trunk:** The primary vertical cross-section of the main stem.
2.  **Branch:** Lateral structural branches extending from the trunk.
3.  **Twigs:** Fine, thin distal branches and terminal growth.
4.  **Grass:** Low-lying ground vegetation and forest floor clutter.

**Annotation Type:** Axis-aligned bounding boxes were drawn around every visible cross-section within the 2D slice.

---

## 2.4 Data Partitioning (Train/Val/Test)
The dataset was split to ensure robust evaluation of the model's ability to generalize to new environments.

| Set | Purpose | Strategy |
| :--- | :--- | :--- |
| **Training** | Model Optimization | 80% of labelled slices from internal sites. |
| **Validation** | Hyperparameter Tuning | 20% of labelled slices; used for mAP metrics. |
| **Test** | Generalization Check | Slices from trees **outside** the training sites. |

### Evaluation Constraints
Due to project time constraints, the **Test Set** remains unlabelled. While the Training and Validation sets provide quantitative metrics (mAP, F1-Score), the final performance on the Test Set is assessed via **visual evaluation** of the segmented point cloud output.