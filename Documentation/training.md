# 5. Training Process

This chapter describes how the adapted YOLO11 model was fine-tuned on the prepared dataset. Training was performed using the **Ultralytics** training engine, which handles the optimization loop, metric tracking, and validation routines. Custom scripts in the project repository were used to launch the training and verify that the modified model correctly accepted the four-channel input tensors.

The goal of this stage was to allow the pretrained detector to learn how to incorporate the additional structural information provided by the extra channel while maintaining stable convergence.

---

## 5.1 Training Configuration

Training was executed on a CUDA-enabled workstation using GPU acceleration through the Ultralytics framework. The training run was launched via the script `train_4ch_detection.py`, which loads the custom model configuration and dataset definition before starting the optimization loop.

Before training begins, the script prints the shape of the first convolutional layer. This verification step confirms that the loaded checkpoint has successfully been converted to accept **four input channels**, ensuring that the modification performed earlier in `make_4ch_checkpoint.py` is active.

The primary training parameters are summarized below.

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Epochs** | 300 | Maximum number of training iterations |
| **Batch Size** | 32 | Balanced GPU memory usage and gradient stability |
| **Image Size** | 640 px | Standard YOLO input resolution |
| **Device** | GPU (CUDA) | Hardware acceleration enabled |
| **AMP** | Enabled | Automatic Mixed Precision for faster training |
| **Optimizer** | Auto | Optimizer automatically selected by Ultralytics |

> [!NOTE] Hardware Considerations  
> The chosen configuration was designed to balance training speed and memory usage on the available GPU. The batch size and image resolution allowed stable training without exceeding VRAM limits.

---

## 5.2 Convergence Behaviour

Although the maximum training duration was set to **300 epochs**, the effective learning phase concluded earlier. The validation metrics stabilized between **Epoch 180 and 200**, indicating that the network had extracted most of the useful patterns from the dataset.

After this point, improvements in the validation metrics were minimal. Continuing the training would therefore increase computation time without producing meaningful performance gains.

> [!TIP] Training Plateau  
> When validation metrics stabilize across multiple epochs, the model has typically reached the performance limit supported by the dataset and model capacity. Stopping near this plateau avoids unnecessary computation while preserving the best-performing weights.

---

## 5.3 Performance Metrics

During training, the Ultralytics engine continuously evaluated the model on the validation subset. Several metrics were monitored to assess both localization accuracy and classification performance.

### Box Loss

This metric measures how accurately the predicted bounding boxes align with the annotated objects. Lower values indicate better localization of tree components within the slice images.

### mAP@50 (Mean Average Precision)

The primary evaluation metric was the **mean average precision at an Intersection over Union (IoU) threshold of 0.5**. This value summarizes how reliably the model detects and classifies objects across the validation dataset.

For this project, the metric mainly reflects how well the model distinguishes between **trunks, branches, twigs, and grass**.

### F1 Score

The **F1 score** combines precision and recall into a single measure. It was particularly useful for evaluating the **twig class**, where sparse point distributions can easily be confused with background noise.

Maintaining a balance between detecting small twig structures and avoiding false positives was therefore an important indicator of model quality.

---

## 5.4 Validation Strategy

The validation set consisted of slices derived from the same trees used for training but representing **different vertical sections**. This ensured that the model was evaluated on images it had not seen before while still preserving realistic structural patterns.

> [!NOTE] Validation Limitations  
> Because the validation slices originate from the same trees as the training set, the model may still encounter familiar structural patterns. A more rigorous evaluation was therefore performed later using an external test dataset.

---

## 5.5 Prediction Workflow

After training, the best-performing checkpoint was used for inference on generated slice images. This step was implemented in the repository script `predict_on_train.py`, which loads the trained model and applies it to a directory containing the multi-channel slice images.

The script exports predictions in two formats:

| Output | Purpose |
| :--- | :--- |
| **Annotated Images** | Visual inspection of detection results |
| **YOLO Label Files** | Bounding box coordinates and confidence scores |

These predictions serve as the input for the later post-processing stage, where the detected objects are mapped back to their corresponding locations in the original **3D point cloud**.

> [!TIP] Why Export YOLO Labels  
> Saving predictions in YOLO format makes it straightforward to reuse the results for downstream processing, evaluation, or reprojection into the 3D coordinate system.

