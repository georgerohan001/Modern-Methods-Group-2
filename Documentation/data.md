# 3. Preprocessing and Slicing Pipeline

This chapter documents the workflow used to transform raw terrestrial laser scanning (TLS) data into the image-based dataset used for model training. The aim of this section is both to explain the reasoning behind the processing steps and to allow the workflow to be reproduced by other researchers.

The overall process converts three dimensional point cloud data into structured two dimensional slices that preserve vertical context. These slices are later annotated and used as input for the detection model.

---

# 3.1 Data Acquisition

The point cloud data used in this project was collected during a field excursion as part of the master's module **“Modern methods of forest and environmental surveying using terrestrial laser scanning and UAVs”**, taught by **Julian Frey**.

The excursion took place on **2 March 2026** in the **Mathislewald forest near Freiburg, Germany**. The approximate coordinates of the study location are:

- **Latitude:** 47.88539° N  
- **Longitude:** 8.08564° E  
- **Coordinate Reference System:** WGS84 (EPSG:4326)

During the excursion, students were introduced to the operation of terrestrial laser scanning devices and UAV-based survey methods. As part of the exercise, a full TLS scan of a forest stand was recorded. The resulting stand-level point cloud dataset was later made available to the students for further analysis and experimentation.

This dataset served as the starting point for the workflow described in this project.

---

# 3.2 Stand-Level Tree Segmentation

The TLS scan contained the entire forest stand in a single point cloud. For the purposes of training a model to recognize structural components of trees, individual trees needed to be isolated from this dataset.

To avoid implementing a stand segmentation algorithm from scratch, the workflow provided by **3dtrees.earth** was used to perform the initial segmentation of trees from the full point cloud. This workflow applies automated tree detection and segmentation methods to separate the stand into individual trees.

The output of this process was a large segmented point cloud in which each point contained an identifier indicating the tree to which it belonged.

Although this solved the segmentation problem, the resulting file was still extremely large and difficult to work with in interactive point cloud software.

---

# 3.3 Splitting the Segmented Dataset

When attempting to load the segmented dataset in **CloudCompare**, the software frequently froze or crashed due to the size of the file.

To make the data manageable, a custom **R script named `Segmentor.R`** was written and executed. The purpose of this script was to separate the segmented point cloud into **one LAS file per tree**.

The script performs the following steps:

1. Load the stand-level TLS point cloud.
2. Classify ground points using the CSF ground classification algorithm.
3. Generate a Digital Terrain Model (DTM).
4. Normalize point heights relative to the terrain.
5. Compute geometric descriptors required by the segmentation algorithm.
6. Detect candidate tree base locations.
7. Perform CSP-COST tree segmentation.
8. Assign a **TreeID** to each point.
9. Split the dataset by TreeID.
10. Export each tree into its own `.las` file.

After running the script, the output directory contained many smaller LAS files named in the format:

```
Tree_001.las
Tree_002.las
Tree_003.las
...
```

This step significantly improved usability. Instead of loading the entire forest stand at once, it became possible to load **20 to 50 trees simultaneously** in CloudCompare while still retaining a visual sense of the surrounding stand.

The script itself is available in the project repository under the filename **`Segmentor.R`**.

---

# 3.4 Manual Inspection and Tree Selection

Once the trees had been separated into individual files, they were inspected in **CloudCompare**.

Automated segmentation is rarely perfect. Some trees were split incorrectly, while others contained points from neighbouring trees or understory vegetation.

Each member of the group therefore selected **two to three trees** that required the least amount of correction. Minor segmentation issues were manually cleaned by removing stray points so that each point cloud represented **a single, coherent tree**.

These cleaned tree point clouds served as the input for the slicing pipeline.

---

# 3.5 Vertical Slicing of Trees

To convert the 3D tree point clouds into a form suitable for computer vision models, the trees were sliced horizontally into thin layers.

This step was implemented using a Python script named **`slicer.py`**, which is also available in the project repository.

The script performs the following operations for each tree:

1. Load the `.las` point cloud.
2. Determine the global bounding box of the tree in the XY plane.
3. Determine the vertical extent of the tree.
4. Divide the tree into horizontal slices with a fixed thickness.

The parameters used in this project were:

| Parameter | Value | Description |
|---|---|---|
| Slice height | 0.20 m | Vertical thickness of each slice |
| Pixel size | 0.01 m | Raster resolution in the XY plane |

For every slice:

1. All points whose height falls within the slice interval are selected.
2. The points are projected onto the XY plane.
3. A **2D histogram** is computed to represent point density within the slice.
4. The density values are normalized and converted to an **8-bit grayscale image**.
5. The resulting raster is saved as a **PNG image**.

Each image therefore represents the **horizontal structure of the tree at a specific height**.

An example filename looks as follows:

```
Tree_012_slice_034.png
```

This naming convention allows the slice index and source tree to be easily identified.

---

# 3.6 Metadata Generation

During slicing, the script also generates a **metadata file** (`metadata.json`) that stores spatial information for every exported slice.

For each image, the metadata records:

- The originating tree
- The global XY origin of the raster
- The vertical position of the slice
- Pixel resolution
- Image dimensions

This information is important because it allows detections made on the images to later be mapped back into the **original three dimensional coordinate system**.

---

# 3.7 Annotation of Tree Components

Once the slice images were generated, they were imported into the annotation platform **CVAT**.

Each image was manually labelled to identify structural components of the tree. The following four classes were defined:

| Label | Description |
|---|---|
| grass | ground vegetation visible in lower slices |
| twigs | very thin woody structures |
| trunk | main stem of the tree |
| branch | larger secondary woody structures |

The annotation process involved drawing bounding boxes around visible structures within each slice.

Although this step was time intensive, it produced a labelled dataset that links point cloud derived imagery to meaningful structural categories.

---

# 3.8 Export to YOLO Dataset Format

After annotation was completed, the dataset was exported from CVAT in **YOLO 1.1 format**.

In this format:

- Each image is paired with a **text file** containing the bounding box annotations.
- Each class label is represented by a **numeric index**.

An additional configuration file defines the mapping between indices and labels, for example:

```
0 grass
1 twigs
2 trunk
3 branch
```

This export produced a dataset that can be directly used by YOLO based detection models.

The preparation of training and validation subsets is described in the following chapter.

---