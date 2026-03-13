# 4. Model Architecture

This section describes the transition from a standard computer vision framework to a customized multi-channel architecture designed for 3D-to-2D forest segmentation.

---

## 4.1 Framework & Base Model
We utilized the **Ultralytics YOLO11** framework, selecting the **YOLO11s (Small)** variant as our core architecture.

### Justification for the "Small" Variant
While larger models (m, l, x) offer higher parameter counts, the **YOLO11s** was chosen for several strategic reasons:
* **Hardware Efficiency:** Enables faster training iterations and lower VRAM usage on available workstation hardware.
* **Feature Complexity:** 2D cross-sections of tree components (circles, ovals, and scattered points) are geometrically simpler than general COCO objects (e.g., "person" or "car"), making the "Small" capacity sufficient for high-accuracy detection.
* **Inference Speed:** Essential for the eventual "re-projection" phase where thousands of slices per tree must be processed.

---

## 4.2 Single-Channel vs. Multi-Channel Configuration

Initially, the model was tested using standard single-channel density slices. However, to leverage the vertical context and height indices described in the Preprocessing section, a **4-channel (RGBA-style) input** was implemented.

### The 4-Channel "Weight Expansion" Hack
Standard pretrained models (like `yolo11s.pt`) are hardcoded for 3-channel RGB input. To utilize 4 channels while retaining the "knowledge" of pretrained COCO weights, we performed a kernel expansion on the first convolutional layer.

**The Workflow:**
1. **Input Modification:** The model and data configurations were set to `channels: 4`.
2. **Weight Initialization:** We developed a custom script (`make_4ch_checkpoint.py`) to modify the initial weight tensor.
3. **Channel Mapping:** * The weights for the first 3 channels (R, G, B) were copied directly from the pretrained weights.
   * The **4th channel** (Height Index) was initialized using the **mean of the original RGB weights**.
   
> [!TIP] Why use the Mean?
> Initializing the 4th channel with the mean of the existing kernels ensures that the new channel starts with a feature-detection "logic" similar to the rest of the network, preventing massive gradient instability during the first few epochs of fine-tuning.

---

## 4.3 Technical Specifications

| Feature | Specification |
| :--- | :--- |
| **Model Variant** | YOLO11s |
| **Input Resolution** | 640 × 640 pixels (Maximum) |
| **Input Channels** | 4 (Density, Slice-1, Slice-2, Height Index) |
| **Total Classes** | 4 (Trunk, Branch, Twigs, Grass) |
| **Pretrained Base** | `yolo11s.pt` (Fine-tuned via custom weights) |

---

## 4.4 Task Definition: Detection vs. Segmentation
Although the final goal is the semantic classification of every point in the `.las` file, we treat the intermediate 2D task as **Object Detection**. 

By predicting bounding boxes for every cross-section, we can:
1. Identify the centroid and extent of each structural element.
2. Filter out noise more effectively than pixel-wise segmentation in sparse "twig" areas.
3. Map these 2D boxes back to 3D cylinders or voxels during the post-processing phase.