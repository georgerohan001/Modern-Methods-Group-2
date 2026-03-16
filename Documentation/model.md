# 2.3 Model Architecture

This chapter documents how the base YOLO11 model was adapted to accept the custom multi-channel inputs and the repository files that contain the exact changes. The text focuses on what was done and why, not on how to run the scripts (the code is available on GitHub).

---

## 2.3.1 Base framework and rationale

The project used the YOLO11s variant from the Ultralytics YOLO11 family as the starting architecture. The small variant was chosen to balance detection capacity with rapid iteration and low inference cost when processing many slices per tree.

---

## 2.3.2 Adapting a pretrained 3-channel model to 4 channels

Pretrained YOLO11 checkpoints are configured for three input channels. To retain pretrained representations while accepting the four-channel inputs produced by the preprocessing pipeline, the first convolutional layer was expanded from 3 → 4 input channels.

The pretrained checkpoint originally contained a first-layer weight tensor with the shape `out_c × 3 × k × k`, corresponding to three input channels. To incorporate an additional height index channel, a new tensor with shape `out_c × 4 × k × k` was created. The pretrained weights for the original three channels were copied directly into the new tensor, while the weights for the fourth channel were initialized using the per-kernel mean of those three existing channels. The modified tensor was then written back into a copy of the checkpoint and saved.

Initializing the new channel with the mean of the existing kernels provides a neutral but informative starting point, which helps avoid the instability that can occur with random initialization. The script that performs this modification is `make_4ch_checkpoint.py`, which is included in the repository.

---

## 2.3.3 Model and dataset configuration files

Two YAML files encapsulate the model and dataset intent:

- `data11.yaml` — model definition where `channels: 4` is set and the backbone/head structure resides.  
- `data.yaml` — dataset descriptor that records `nc`, `names`, and `channels: 4`.

These files ensure the model is constructed with `ch=4` and that class indices are consistent across training and postprocessing.

A minimal local edit was made to the training entry point so the `channels` value from the dataset config is honored during model construction rather than being silently overridden. The code change reads the `channels` field and passes it into the `DetectionModel` constructor.