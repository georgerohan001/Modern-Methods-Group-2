# 5. Training Process

The training phase was executed using the **Ultralytics** engine, specifically optimized for our custom 4-channel input tensors. This section details the hyperparameter configuration and the convergence behavior of the model.

---

## 5.1 Training Configuration
The model was fine-tuned on a high-performance workstation using a CUDA-enabled GPU. The following hyperparameters were used to ensure stable convergence of the weights:

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Epochs** | 300 | Maximum iteration limit. |
| **Batch Size** | 32 | Balanced for VRAM efficiency and gradient stability. |
| **Image Size** | 640 px | Standard resolution for YOLOv11 high-fidelity detection. |
| **Device** | GPU (0) | CUDA acceleration enabled. |
| **AMP** | True | Automatic Mixed Precision for faster training. |
| **Optimizer** | Auto | Ultralytics default (typically SGD or AdamW). |

---

## 5.2 Convergence & Early Termination
While the training was initialized for 300 epochs, the model utilized **Early Stopping** logic to prevent overfitting and save computational resources.

* **Termination Point:** Training effectively concluded between **Epoch 180 and 200**.
* **Reasoning:** The validation loss and mAP (Mean Average Precision) metrics reached a plateau. Improvements in subsequent epochs were statistically insignificant, indicating that the model had reached its optimal capacity for the provided dataset.

---

## 5.3 Performance Metrics
During the training process, the following key metrics were monitored via the validation set (20% split):

### 1. Box Loss
Measures how accurately the model locates the center of a tree component and how well the predicted bounding box covers the object.

### 2. mAP@50 (Mean Average Precision)
This represents the accuracy of the model at an Intersection over Union (IoU) threshold of 0.5. It is the primary indicator of the model's ability to correctly identify **Trunks** vs. **Branches**.

### 3. F1-Score
The harmonic mean of Precision and Recall. This was particularly important for our **"Twigs"** class, where we needed to balance the model's sensitivity (finding all twigs) with its precision (not misidentifying noise as twigs).

> [!NOTE] Validation Strategy
> Because the validation set included images from the same trees used in training (but different slices), the model showed high proficiency in recognizing familiar structural patterns. The true test of robustness was reserved for the unlabelled external test site.

---

## 5.4 Monitoring the First Convolution
Before the training loop commenced, a verification check was performed on the **First Convolutional Layer**. 

By confirming the weight shape as `[Out, 4, K, K]`, we ensured that the `make_4ch_checkpoint.py` modification had successfully integrated the 4th "Height Index" channel into the pretrained architecture without breaking the weight loading process.