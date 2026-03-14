# 2. Data Acquisition and Preparation

This chapter describes the complete workflow used to transform raw terrestrial laser scanning (TLS) data into a labeled image dataset suitable for training a YOLO-based object detection model. The process consists of four main stages: field data acquisition, tree segmentation, preparation of individual tree point clouds, and the conversion of three-dimensional data into two-dimensional raster images suitable for annotation.

The steps described below are intended not only to document the workflow used in this project, but also to allow the process to be reproduced by other researchers.

---

## 2.1 Field Data Acquisition

The raw data used in this study were collected during a field excursion as part of the Master's module **“Modern methods of forest and environmental surveying using terrestrial laser scanning and UAVs”**, led by **Julian Frey** at the University of Freiburg.

The field campaign took place on **2 March 2026** in the **Mathislewald**, located in the Black Forest near Freiburg, Germany. The approximate coordinates of the study site are:

- **WGS84 (DMS):** 47°53'07.4"N, 8°05'08.3"E  
- **WGS84 (Decimal):** 47.88539° N, 8.08564° E  
- **Projected CRS:** UTM Zone 32N (EPSG:32632)

During the excursion, students were trained in the operation of both **terrestrial laser scanning (TLS)** systems and **UAV-based surveying platforms**. The TLS data collected during this exercise were subsequently made available to participants for use in individual analysis projects.

The point clouds used in this work were captured using a **RIEGL VZ-400i terrestrial laser scanner**, a high-resolution TLS instrument capable of producing dense three-dimensional representations of forest stands.

The scanned stand consists of a **mixed coniferous forest**, predominantly composed of *Picea abies* (Norway spruce), along with smaller understory vegetation.

---

## 2.2 Tree Segmentation from the Stand Point Cloud

The TLS scans initially represent the entire forest stand as a single large point cloud. For the purposes of this project, however, individual trees were required as separate datasets.

To obtain these, the stand-level point cloud was processed using the segmentation workflow provided by **3dtrees.earth**. This platform implements automated tree segmentation methods designed for terrestrial laser scanning data and allows individual trees to be extracted from a larger forest scan.

Using this workflow allowed the project to focus on later processing stages rather than developing a custom tree segmentation pipeline from scratch.

The output of this step was a **large segmented point cloud file** in which each point was assigned a **TreeID**, identifying the tree to which the point belongs.

---

## 2.3 Splitting the Segmented Dataset into Individual Tree Files

The segmented dataset contained all trees within a single `.las` file. Due to the large size of this dataset, loading the entire file into **CloudCompare** frequently resulted in freezing or crashing of the software.

To make the dataset manageable, the file was divided into separate point clouds containing **one tree per file**. This was done using a custom R script called **`Segmentor.R`**, which has been made available in the project repository.

The script uses the **lidR** ecosystem for point cloud processing and performs the following operations:

1. **Loading the TLS dataset**  
   The stand-level `.las` file is read into R using `lidR`.

2. **Ground classification**  
   Ground points are identified using the **Cloth Simulation Filter (CSF)** algorithm.

3. **Digital Terrain Model (DTM) generation**  
   A terrain surface is created using a **Triangulated Irregular Network (TIN)** interpolation.

4. **Height normalization**  
   The terrain model is subtracted from the point cloud so that all point heights represent **height above ground** rather than absolute elevation.

5. **Removal of ground-level noise**  
   Points below a minimum height threshold are removed to eliminate residual ground artifacts.

6. **Computation of geometric descriptors**  
   Additional geometric features required by the segmentation algorithm are calculated.

7. **Tree segmentation**  
   The **CSP-COST segmentation algorithm** is applied to assign each point a **TreeID**.

8. **Export of individual trees**  
   The final step splits the dataset according to the TreeID attribute and exports **one `.las` file per tree**.

The primary purpose of this script was therefore not to perform the initial segmentation, but to **divide the segmented dataset into manageable files**, enabling interactive inspection and correction of individual trees in CloudCompare.

After running the script, each tree existed as a separate `.las` file, which could be loaded independently.

---

## 2.4 Manual Inspection and Selection of Trees

Once the individual tree files were created, they were loaded into **CloudCompare** for visual inspection.

Although the automated segmentation performed reasonably well, some trees contained artifacts such as:

- fragments of neighboring trees  
- incorrectly assigned branch points  
- residual background points

To minimize noise in the training dataset, each group member selected **two to three trees** that required the least amount of manual correction.

Minor adjustments were then made within CloudCompare to remove misclassified points and ensure that each dataset represented **a single clean tree point cloud**.

These corrected trees formed the basis for the subsequent slicing workflow.

---

## 2.5 Converting 3D Trees into 2D Image Slices

Object detection models such as YOLO operate on **two-dimensional images**, while TLS data consists of **three-dimensional point clouds**. To bridge this gap, each tree was converted into a stack of two-dimensional slices.

This conversion was performed using a Python script called **`slicer.py`**, also available in the project repository.

### Slicing Strategy

The script processes each tree point cloud as follows:

1. **Loading the point cloud**  
   Each `.las` file is read using the `laspy` library.

2. **Determining spatial bounds**  
   The global X and Y extent of the tree is calculated to define the image canvas.

3. **Vertical stratification**  
   The tree is divided into horizontal layers with a thickness of **20 cm** along the Z-axis.

4. **Slice extraction**  
   For each layer, all points whose height falls within the slice are selected.

5. **Projection to 2D**  
   The selected points are projected onto the **XY plane**, effectively collapsing the slice into two dimensions.

6. **Density rasterization**  
   A **2D histogram** is computed to represent the spatial density of points within the slice.

7. **Spatial resolution**  
   The raster grid uses a resolution of **1 cm per pixel**, producing a detailed representation of each cross-section.

8. **Intensity normalization**  
   To avoid extreme density values dominating the image, the pixel intensities are clipped at the **99th percentile** and rescaled to **0–255 (uint8)**.

9. **Image export**  
   Each slice is saved as a **PNG image**.

All slices belonging to a given tree share identical spatial bounds, ensuring that the relative position of structures remains consistent across the vertical sequence.

In addition to the images, the script also generates a **metadata file (`metadata.json`)** containing information such as:

- slice height  
- global spatial origin  
- pixel resolution  
- canvas dimensions

This metadata enables the original 3D spatial context to be reconstructed if required.

On average, this process generated approximately **150 slices per tree**, depending on tree height.

---

## 2.6 Dataset Annotation

The resulting slice images were imported into **CVAT (Computer Vision Annotation Tool)** for manual labeling.

Each image was inspected and annotated using **axis-aligned bounding boxes** to identify visible structural elements within the slice.

Four object classes were defined:

| Class | Description |
|------|-------------|
| **Trunk** | Cross-sections belonging to the main stem |
| **Branch** | Larger lateral branches extending from the trunk |
| **Twigs** | Fine distal branches and terminal growth |
| **Grass** | Low vegetation and forest floor clutter |

The annotation process required careful visual interpretation, since individual slices represent only a thin horizontal cross-section of the tree.

After completion of the labeling process, the dataset was exported from CVAT in **YOLO 1.1 format**. In this format:

- Each image is associated with a **text file containing bounding box coordinates**.
- Each object class is represented by a **numeric index (0–3)**.
- A separate configuration file maps each index to its corresponding class label.

This labeled dataset forms the foundation for training the object detection model described in the following chapters.