# 1. Motivation

The transition from traditional forest inventory to high-precision individual tree modeling represents a paradigm shift in sustainable forest management. As foresters move beyond stand-level estimates, the ability to decompose complex point clouds into structural components becomes the foundational step for advanced analytics.

---

## The Necessity of Structural Decomposition
Decomposing a single-tree point cloud into its primary components—**trunk, branches, twigs, and ground/grass**—is a prerequisite for several high-value forestry applications:

* **Trunk Volume & Taper:** Accurate estimation of merchantable volume and stem tapering.
* **Wood Quality Metrics:** Assessing stem straightness and potential bending stress.
* **Biomass Modeling:** Moving beyond allometric equations to direct geometric measurements.
* **Structural Analysis:** Understanding tree architecture for ecological and physiological studies.

> [!IMPORTANT] Our Approach
> We transform the complex 3D detection problem into a streamlined **2D object detection** task. By horizontally slicing the point cloud and generating 2D density rasters, we leverage the speed and accuracy of **YOLOv11s** on multichannel cross-sectional images.

---

## Background & Literature Review

### From Stand-Level to Individual Trees
Forestry is increasingly utilizing 3D measurements to move beyond broad stand-level estimates toward models of individual trees (Calders et al., 2017). A critical step in this process is breaking a single-tree point cloud into its main structural parts—specifically separating the stem from branches, twigs, and ground points. This distinction is vital because accurate stem isolation directly improves estimates of taper, trunk volume, and stem quality derived from LiDAR data (Chen et al., 2024; Luoma et al., 2022).

### Limitations of Traditional Geometric Fitting
Traditionally, stem biomass and volume have been estimated using allometric equations based on manual measurements like DBH (Diameter at Breast Height). Modern methods utilize LiDAR-based **Quantitative Structure Models (QSMs)**, such as *SimpleTree* and *AdQSM*, which fit cylinders directly to the tree’s geometry (Hackenberg et al., 2015; Raumonen et al., 2020; Pérez-Cruz et al., 2022).

However, these cylinder-fitting methods often struggle with:
1.  **Occlusion:** Missing data points in dense canopies.
2.  **Clutter:** Noisy points from surrounding vegetation.
3.  **Bias:** Branches or twigs being misidentified as part of the stem, distorting volume estimates (Hackenberg et al., 2015).

### The YOLOv11 Workflow
To overcome these limitations, we treat structural decomposition as a **2D object-detection task**. By projecting horizontal slices of the point cloud into multi-channel density images, we use **YOLOv11s** to detect cross-sectional structures. This isolates the stem before further geometric modeling occurs, creating a cleaner input for biomass and taper estimation while avoiding the high computational overhead of full **3D semantic segmentation** (Hao et al., 2022; Liu et al., 2024; Wang et al., 2024).

---

## References
* **Calders, K., et al. (2017).** *Realistic terrestrial laser scanning simulations of forest canopies.*
* **Chen, Z., et al. (2024).** *Advancements in LiDAR-based individual tree modeling.*
* **Hackenberg, J., et al. (2015).** *SimpleTree — An Efficient Open Source Tool to Estimate Tree Structural Parameters and Biomass from TLS Data.*
* **Hao, Z., et al. (2022).** *Deep Learning in Forest Point Cloud Segmentation.*
* **Luoma, V., et al. (2022).** *Stem quality assessment using terrestrial laser scanning.*
* **Raumonen, P., et al. (2020).** *Quantitative structure models for individual trees.*
* **Wang, Y., et al. (2024).** *Efficiency of 2D vs 3D Neural Networks in Forestry Applications.*