# 4. Model Architecture

This chapter documents how the base YOLO11 model was adapted to accept the custom multi-channel inputs and the repository files that contain the exact changes. The text focuses on what was done and why, not on how to run the scripts (the code is available on GitHub).

---

## 4.1 Base framework and rationale

The project used the YOLO11s variant from the Ultralytics YOLO11 family as the starting architecture. The small variant was chosen to balance detection capacity with rapid iteration and low inference cost when processing many slices per tree.

Key practical reasons:
- reduced VRAM and faster training/inference,
- sufficient capacity for the geometric patterns present in slice images,
- lower computational cost for the 2D→3D reprojection stage.

---

## 4.2 Adapting a pretrained 3-channel model to 4 channels

Pretrained YOLO11 checkpoints are configured for three input channels. To retain pretrained representations while accepting the four-channel inputs produced by the preprocessing pipeline, the first convolutional layer was expanded from 3 → 4 input channels.

Conceptual summary of the adaptation:
- the original first-layer weight tensor (shape: `out_c × 3 × k × k`) was read from the checkpoint,
- a new tensor with shape `out_c × 4 × k × k` was allocated,
- the pretrained weights for the first three channels were copied unchanged,
- the new fourth channel (height index) was initialized as the per-kernel mean of the original three channels,
- the modified tensor was written back into a copied checkpoint and saved.

Rationale for the initialization choice:
- using the mean provides a neutral, informative starting point for the new channel and reduces early gradient instability compared with random initialization,
- this preserves learned edge/texture detectors while allowing the network to integrate the additional structural cue.

The script that performs this operation is named `make_4ch_checkpoint.py` and is tracked in the repository.

---

## 4.3 Model and dataset configuration files

Two YAML files encapsulate the model and dataset intent:

- `data11.yaml` — model definition where `channels: 4` is set and the backbone/head structure resides.  
- `data.yaml` — dataset descriptor that records `nc`, `names`, and `channels: 4`.

These files ensure the model is constructed with `ch=4` and that class indices are consistent across training and postprocessing.

A minimal local edit was made to the training entry point so the `channels` value from the dataset config is honored during model construction rather than being silently overridden. The code change reads the `channels` field and passes it into the `DetectionModel` constructor.

---

## 4.4 Technical summary

| Item | Value |
|---:|:---|
| Model variant | YOLO11s |
| Input channels | 4 (as produced by preprocessing) |
| Input resolution | up to 640 × 640 px (project-level choice) |
| Classes | 4 (twigs, trunk, branch, grass) |
| Pretrained base | `yolo11s.pt` → converted with `make_4ch_checkpoint.py` |

Note: the model still uses standard detection heads (P3/P4/P5). During inference, 2D detections are reprojected into the original 3D frame using the slice metadata (this reprojection is handled by the project’s postprocessing scripts).

---