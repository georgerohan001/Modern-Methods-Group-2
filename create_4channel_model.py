"""Create a 4-channel YOLO11s model from pretrained yolo11s.pt."""

import torch
from pathlib import Path
from ultralytics import YOLO
from datetime import datetime


def create_4channel_model(
    input_model: str = "yolo11s.pt",
    output_model: str = "yolo11s_4ch.pt",
    num_classes: int = 4,
):
    print(f"Loading pretrained model: {input_model}")

    ckpt = torch.load(input_model, map_location="cpu", weights_only=False)
    print(f"Checkpoint keys: {ckpt.keys()}")

    model = ckpt.get("model", ckpt.get("ema", None))
    if model is None:
        print("Could not find model in checkpoint")
        return None

    print(f"Original first conv in_channels: {model.model[0].conv.in_channels}")

    first_conv = model.model[0].conv

    new_conv = torch.nn.Conv2d(
        in_channels=4,
        out_channels=first_conv.out_channels,
        kernel_size=first_conv.kernel_size,
        stride=first_conv.stride,
        padding=first_conv.padding,
        bias=first_conv.bias is not None,
    )

    with torch.no_grad():
        new_conv.weight[:, :3, :, :] = first_conv.weight
        new_conv.weight[:, 3:4, :, :] = first_conv.weight.mean(dim=1, keepdim=True)
        if first_conv.bias is not None:
            new_conv.bias = torch.nn.Parameter(first_conv.bias.data.clone())

    model.model[0].conv = new_conv

    print(f"Modified first conv in_channels: {model.model[0].conv.in_channels}")

    new_ckpt = {
        "date": datetime.now().isoformat(),
        "version": "11.0.0",
        "license": "AGPL-3.0",
        "docs": "",
        "epoch": 0,
        "best_fitness": 0.0,
        "model": model,
        "ema": None,
        "updates": None,
        "optimizer": None,
        "train_args": {
            "data": "",
            "epochs": 100,
            "batch": 16,
            "imgsz": 640,
            "nc": num_classes,
            "names": ["twigs", "trunk", "branch", "grass"],
        },
        "train_metrics": None,
        "train_results": None,
    }

    save_path = Path(output_model)
    torch.save(new_ckpt, save_path)
    print(f"Saved 4-channel model to: {save_path}")

    return str(save_path)


def create_4channel_model_from_yaml(
    output_model: str = "yolo11s_4ch.pt",
    num_classes: int = 4,
):
    print("Creating 4-channel YOLO11s model from YAML config")

    import yaml
    import tempfile

    custom_yaml = {
        "nc": num_classes,
        "scales": {
            "n": [0.5, 0.25, 1024],
            "s": [0.5, 0.5, 1024],
            "m": [0.5, 1.0, 512],
            "l": [1.0, 1.0, 512],
            "x": [1.0, 1.5, 512],
        },
        "backbone": [
            [-1, 1, "Conv", [64, 3, 2]],
            [-1, 1, "Conv", [128, 3, 2]],
            [-1, 2, "C3k2", [256, False, 0.25]],
            [-1, 1, "Conv", [256, 3, 2]],
            [-1, 2, "C3k2", [512, False, 0.25]],
            [-1, 1, "Conv", [512, 3, 2]],
            [-1, 2, "C3k2", [512, True]],
            [-1, 1, "Conv", [1024, 3, 2]],
            [-1, 2, "C3k2", [1024, True]],
            [-1, 1, "SPPF", [1024, 5]],
            [-1, 2, "C2PSA", [1024]],
        ],
        "head": [
            [-1, 1, "nn.Upsample", ["None", 2, "nearest"]],
            [[-1, 6], 1, "Concat", [1]],
            [-1, 2, "C3k2", [512, False]],
            [-1, 1, "nn.Upsample", ["None", 2, "nearest"]],
            [[-1, 4], 1, "Concat", [1]],
            [-1, 2, "C3k2", [256, False]],
            [-1, 1, "Conv", [256, 3, 2]],
            [[-1, 13], 1, "Concat", [1]],
            [-1, 2, "C3k2", [512, False]],
            [-1, 1, "Conv", [512, 3, 2]],
            [[-1, 10], 1, "Concat", [1]],
            [-1, 2, "C3k2", [1024, True]],
            [[16, 19, 22], 1, "Detect", ["nc"]],
        ],
        "ch": 4,
        "scale": "s",
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(custom_yaml, f)
        temp_yaml = f.name

    try:
        model = YOLO(temp_yaml)
        print(
            f"Created model with first conv in_channels: {model.model.model[0].conv.in_channels}"
        )
    except Exception as e:
        print(f"Error creating from YAML: {e}")
        return None
    finally:
        import os

        os.unlink(temp_yaml)

    save_path = Path(output_model)
    torch.save(model.model.state_dict(), save_path)
    print(f"Saved 4-channel model to: {save_path}")

    return str(save_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create 4-channel YOLO11 model")
    parser.add_argument(
        "--input", "-i", default="yolo11s.pt", help="Input model (default: yolo11s.pt)"
    )
    parser.add_argument(
        "--output",
        "-o",
        default="yolo11s_4ch.pt",
        help="Output model (default: yolo11s_4ch.pt)",
    )
    parser.add_argument(
        "--nc", type=int, default=4, help="Number of classes (default: 4)"
    )
    parser.add_argument(
        "--from-yaml",
        action="store_true",
        help="Create from YAML instead of loading pretrained weights",
    )

    args = parser.parse_args()

    if args.from_yaml:
        create_4channel_model_from_yaml(args.output, args.nc)
    else:
        create_4channel_model(args.input, args.output, args.nc)
