"""Create multi-channel TIFFs from tree slice images for YOLO11."""

import re
import sys
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm


def load_image(path: Path) -> np.ndarray:
    img = Image.open(path)
    if img.mode != "L":
        img = img.convert("L")
    return np.array(img, dtype=np.uint8)


def get_tree_prefix_and_slice(filename: str):
    patterns = [
        r"^(.+?)_slice_(\d+)",  # bea_tree_slice_000.png, tree_00975_slice_000.png
        r"^(.+?)_(\d{3})\.",  # tree_00638_slice_000.png
    ]
    for pattern in patterns:
        m = re.match(pattern, filename)
        if m:
            prefix = m.group(1)
            slice_num = int(m.group(2))
            return prefix, slice_num
    return None, -1


def create_multichannel_tiffs(input_dir: Path, output_dir: Path, dataset_name: str):
    print(f"\n{'=' * 50}")
    print(f"Processing: {dataset_name}")
    print(f"{'=' * 50}")

    if not input_dir.exists():
        print(f"ERROR: Input dir not found: {input_dir}")
        return False

    train_dir = input_dir / "train"
    if not train_dir.exists():
        print(f"ERROR: train dir not found: {train_dir}")
        return False

    train_files = sorted(train_dir.glob("*.png"))
    if not train_files:
        print(f"ERROR: No images in {train_dir}")
        return False

    # Group files by tree prefix
    tree_groups = {}
    for f in train_files:
        prefix, slice_num = get_tree_prefix_and_slice(f.name)
        if prefix is None:
            continue
        if prefix not in tree_groups:
            tree_groups[prefix] = {}
        tree_groups[prefix][slice_num] = f

    if not tree_groups:
        print(f"ERROR: Could not parse filenames")
        return False

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Input:  {train_dir}")
    print(f"Output: {output_dir}")
    print(f"Total files: {len(train_files)}")
    print(f"Tree groups: {list(tree_groups.keys())}")

    total_created = 0

    for tree_prefix, files_by_slice in tree_groups.items():
        slices = sorted(files_by_slice.keys())
        min_slice = min(slices)
        max_slice = max(slices)

        for slice_num in tqdm(slices, desc=f"{tree_prefix}"):
            train_path = files_by_slice[slice_num]

            ch0 = load_image(train_path)

            if slice_num + 1 in files_by_slice:
                ch1 = load_image(files_by_slice[slice_num + 1])
            else:
                ch1 = np.zeros_like(ch0)

            if slice_num + 2 in files_by_slice:
                ch2 = load_image(files_by_slice[slice_num + 2])
            else:
                ch2 = np.zeros_like(ch0)

            ch3 = np.full_like(ch0, slice_num)

            stacked = np.stack([ch0, ch1, ch2, ch3], axis=2)

            output_path = output_dir / f"{train_path.stem}.tiff"
            img = Image.fromarray(stacked, mode="RGBA")
            img.save(output_path, compression="tiff_deflate")
            total_created += 1

    print(f"Done! Created {total_created} TIFFs in {output_dir}")
    return True


def main():
    BASE_DIR = Path(__file__).parent.resolve()

    datasets = [
        (
            "tree_0638",
            BASE_DIR / "data/datasets/datasets/tree_0638/images",
            BASE_DIR / "data/datasets/datasets/tree_0638/multichannel",
        ),
        (
            "my_yolo_dataset",
            BASE_DIR / "data/datasets/datasets/my_yolo_dataset/images",
            BASE_DIR / "data/datasets/datasets/my_yolo_dataset/multichannel",
        ),
    ]

    print("Multi-Channel TIFF Creator")
    print("Creates 4-channel TIFFs: [primary, i+1, i+2, index-gray]")

    for dataset_name, input_dir, output_dir in datasets:
        success = create_multichannel_tiffs(input_dir, output_dir, dataset_name)
        if not success:
            print(f"Failed to process {dataset_name}")

    print("\nAll datasets processed!")


if __name__ == "__main__":
    main()
