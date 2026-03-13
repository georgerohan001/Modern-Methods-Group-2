# 3. Preprocessing & Slicing Pipeline

The core of our methodology lies in the transformation of raw spatial coordinates into informative multi-channel feature maps. By engineering a 4-channel input, we provide the YOLOv11 model with the vertical and contextual cues necessary to distinguish complex forest structures.

---

## 3.1 Multi-Channel Feature Engineering

Unlike standard computer vision tasks that use Red, Green, and Blue channels, our pipeline utilizes a custom **4-channel architecture**. Each channel is designed to encode a specific spatial or contextual dimension of the tree's architecture.

### Channel Composition

The resulting input to the model is a **shape (H, W, 4)** tensor, exported as a 4-channel TIFF (RGBA mode) with `tiff_deflate` compression.

| Channel | Name | Content Description |
| --- | --- | --- |
| **0** | **Primary Slice** | The point density raster of the current slice (i). |
| **1** | **Previous Slice (i-1)** | The density raster of the slice directly below the current one. |
| **2** | **Antecedent Slice (i-2)** | The density raster two levels below the current one. |
| **3** | **Index-Gray** | An opacity gradient representing the relative height (%) of the slice within the total tree height. |

---

## 3.2 Spatial Logic & Vertical Context

The inclusion of channels 1, 2, and 3 is critical for overcoming the limitations of 2D detection in a 3D environment:

* **Vertical Correlation (Channels 1 & 2):** Tree structures like the trunk and primary branches are continuous. Because these features taper upward, providing the model with the "history" of the slices below allows it to better predict the current position and orientation of the stem.
* **Absolute Position (Channel 3):** By encoding the height as a grayscale gradient (black at the base, white at the crown), we provide the model with a "global coordinate." This helps the network distinguish between a wide trunk base (near the ground) and a dense crown (near the top), which might otherwise look similar in a single isolated density slice.

> [!TIP] Handling Edge Cases
> At the very bottom of the tree (where i=0 or i=1), there are no "neighboring" slices below. In these instances, channels 1 and 2 are filled with **zeros (black channels)** to maintain a consistent input shape without introducing noise.

---

## 3.3 Normalization & Rasterization

To ensure the model training is stable, the point density data undergoes the following transformations:

1. **Density Mapping:** Points are binned into 1 cm pixels on the XY plane.
2. **Outlier Clipping:** Values are clipped at the 99th percentile to prevent high-intensity "hotspots" (areas with extreme point density) from washing out the structural details of thinner branches.
3. **8-bit Scaling:** The resulting values are mapped to a **0–255 (uint8)** range.
4. **Spatial Consistency:** All slices within a single tree are processed using the same global bounding box, ensuring that a trunk at coordinates (x, y) in Slice 10 is in the exact same pixel location in Slice 11.

---

### Implementation Note

For the final segmentation, these 4-channel images are passed through the YOLOv11s inference engine. The detected bounding boxes are then "re-projected" back into the original 3D coordinate space by mapping the 2D pixel coordinates back to the Z interval of the specific slice.