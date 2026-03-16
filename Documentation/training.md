# 2.4 Training Process

This chapter describes how the adapted YOLO11 model was fine-tuned on the prepared dataset. Training was performed using the **Ultralytics** training engine, where custom scripts were used to launch the training and verify that the modified model correctly accepted the four-channel input tensors.

The goal was to allow the pretrained detector to learn how to incorporate the additional structural information provided by the extra channel while maintaining stable convergence.

---

## 2.4.1 Training Configuration

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


>[!NOTE]
>Although the maximum training duration was set to **300 epochs**, the effective learning phase concluded earlier. The validation metrics stabilized between **Epoch 180 and 200**, indicating that the network had extracted most of the useful patterns from the dataset, thereby avoiding unnecessary computation while preserving the best-performing weights.

---

## 2.4.2 Validation Strategy

The validation set consisted of slices derived from the same trees used for training but representing **different vertical sections**. This accounted for 20% of the entire dataset. Doing this ensured that the model was evaluated on images it had not seen before while still preserving realistic structural patterns.

> [!NOTE] Validation Limitations  
> Because the validation slices originate from the same trees as the training set, the model may still encounter familiar structural patterns. A more rigorous evaluation was therefore performed later using an external test dataset.

