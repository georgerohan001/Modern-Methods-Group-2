"""Create multi-channel TIFFs from tree slice images for YOLO11."""

import re
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm

BASE_DIR = Path(__file__).parent.resolve()
INPUT_DIR = BASE_DIR / "data/datasets/datasets/tree_0638/images"
OUTPUT_DIR = BASE_DIR / "data/datasets/datasets/tree_0638/multichannel"


def load_image(path: Path) -> np.ndarray:
    img = Image.open(path)
    if img.mode != "L":
        img = img.convert("L")
    return np.array(img, dtype=np.uint8)


def get_slice_num(filename: str) -> int:
    m = re.search(r"_(\d{3})\.", filename)
    return int(m.group(1)) if m else -1


def main():
    print("Multi-Channel TIFF Creator")
    print("=" * 40)

    if not INPUT_DIR.exists():
        print(f"ERROR: Input dir not found: {INPUT_DIR}")
        return

    train_dir = INPUT_DIR / "train"
    train_files = sorted(train_dir.glob("*.png"))

    if not train_files:
        print("ERROR: No images in train/")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Input:  {train_dir}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Files:  {len(train_files)}")

    # Build lookup by slice number
    files_by_slice = {get_slice_num(f.name): f for f in train_files}
    max_slice = max(files_by_slice.keys())

    for train_path in tqdm(train_files, desc="Creating channels"):
        slice_num = get_slice_num(train_path.name)
        if slice_num < 0:
            continue

        # Channel 0: Primary image (H, W)
        ch0 = load_image(train_path)

        # Channel 1: Next slice (i+1)
        if slice_num + 1 <= max_slice and (slice_num + 1) in files_by_slice:
            ch1 = load_image(files_by_slice[slice_num + 1])
        else:
            ch1 = np.zeros_like(ch0)

        # Channel 2: Slice i+2
        if slice_num + 2 <= max_slice and (slice_num + 2) in files_by_slice:
            ch2 = load_image(files_by_slice[slice_num + 2])
        else:
            ch2 = np.zeros_like(ch0)

        # Channel 3: Index-gray (slice number as grayscale value)
        ch3 = np.full_like(ch0, slice_num)

        # Stack as (channels, height, width) for YOLO11
        stacked = np.stack([ch0, ch1, ch2, ch3], axis=0)

        # Save as multi-channel TIFF
        output_path = OUTPUT_DIR / f"{train_path.stem}.tiff"
        # Use Pillow to save multi-channel TIFF
        Image.fromarray(stacked).save(output_path, compression="tiff_deflate")

    print(f"Done! Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
