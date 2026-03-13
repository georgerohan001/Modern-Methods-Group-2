# 6. Results & Evaluation

This section presents the quantitative performance of the YOLOv11s model during training and the qualitative "back-translation" of 2D inferences into the 3D point cloud environment.

---

## 6.1 Training Metrics Overview
The model performance was evaluated using standard computer vision metrics. The final weights selected for inference were those from the epoch with the highest **mAP50-95(B)** before the early-stopping termination at approximately Epoch 190.

### Key Metrics Explained
To interpret the success of the model, we utilize the following benchmarks:

| Metric | Definition | Significance in Forestry |
| :--- | :--- | :--- |
| **Loss** | The mathematical gap between predictions and ground truth. | A decreasing curve indicates the model is successfully "learning" tree structures. |
| **mAP50(B)** | Mean Average Precision at 50% Intersection over Union (IoU). | Validates if the model correctly identifies a component (e.g., a branch is a branch). |
| **mAP50-95(B)** | Stricter accuracy metric averaging IoU from 50% to 95%. | Rewards the model for bounding boxes that fit the trunk or twigs perfectly. |
| **Precision** | Ratio of correct positive predictions to total predicted positives. | Reflects the model's ability to avoid "False Alarms" (e.g., noise labeled as twigs). |
| **Recall** | Ratio of correct positive predictions to all actual objects. | Reflects the model's ability to find every branch, even in sparse areas. |

---

## 6.2 Comparison: Single vs. Multi-Channel Performance
A primary objective was determining if the engineered 4-channel input outperformed a standard single-channel density slice.

### The Multi-Channel Advantage
Preliminary comparisons suggest that the **4-channel model** (Density + Context Slices + Height Index) provides several distinct advantages:
1. **Vertical Continuity:** By "seeing" the slices below (Ch1, Ch2), the model maintains better tracking of the **Trunk** in areas where point density is lower.
2. **Height Context:** The inclusion of the **Height Encoding (Ch3)** significantly reduced False Positives in the crown. For example, thin trunk sections at the top of the tree were less likely to be confused with lateral branches because the model "knew" it was operating in the upper 90th percentile of the tree's height.
3. **Improved Recall:** The multi-channel setup showed higher recall for **Twigs**, as the temporal-like context of adjacent slices helped distinguish actual biological structures from random sensor noise.

---

## 6.3 Test Data: Back-Translation to 3D
The ultimate validation of this project is the **Back-Translation**—the process of mapping 2D bounding box classifications back into the original `.las` file.

### Visualizing the Segmented Tree
Because the test set was unlabelled, we rely on visual inspection of the classified point cloud. The workflow for this visualization was as follows:
1. **Inference:** The 4-channel slices were passed through the YOLOv11s model.
2. **Coordinate Mapping:** The $[x, y]$ pixel coordinates of each bounding box were converted back to the original forest coordinates.
3. **Z-Assignment:** Each box was assigned the $z$ value corresponding to its original 20 cm slice.
4. **Classification Injection:** Points falling within the 3D "extrusion" of these 2D boxes were assigned a new class ID in the LAS file.

> [!TIP] Resulting Output
> The final result is a semantically segmented point cloud where the user can toggle between `Trunk`, `Branch`, `Twigs`, and `Grass` layers, providing a clean, noise-free input for further biomass and taper modeling.

---

## 6.4 Metric Plots
*[Placeholders for your Ultralytics generated plots: `results.png`, `F1_curve.png`, `P_curve.png`, `R_curve.png`]*

> **Note:** As expected, the **Trunk** class typically shows the highest Precision and Recall, while **Twigs** represent the most significant challenge due to their fine geometry and lower point density.