#!/usr/bin/env python3
"""
create_multichannel_tiffs.py
---------------------------------
Create 4‑channel TIFFs (primary slice + next‑2 slices + slice‑index channel)
for a **single** YOLO‑11 dataset – in your case `my_yolo_dataset`.

Folder layout expected (relative to the location of *this* script):

    <project_root>/
        create_multichannel_tiffs.py          ← (this file)
        data/
            datasets/
                my_yolo_dataset/
                    images/                ← *.png slice files
                    multichannel/          ← will be created automatically
"""

# ----------------------------------------------------------------------
# ── Imports ─────────────────────────────────────────────────────────────
# ----------------------------------------------------------------------
import re
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm

# ----------------------------------------------------------------------
# ── Configuration ────────────────────────────────────────────────────
# ----------------------------------------------------------------------
# ── Edit ONLY these two variables if you want to place the data somewhere
#    else on disk.  They are **absolute** paths once resolved.
BASE_DIR = Path(__file__).parent.resolve()                # folder that holds this script
DATA_ROOT = BASE_DIR / "data" / "datasets"                # common parent for all datasets

# Name of the dataset you actually have (no “tree_0638” any more)
DATASET_NAME = "my_yolo_dataset"

# ----------------------------------------------------------------------
# ── Helper functions ─────────────────────────────────────────────────
# ----------------------------------------------------------------------
def get_tree_prefix_and_slice(filename: str):
    """
    Extracts:
        * prefix – groups all slices that belong to the same tree / volume
        * slice number – integer that defines the ordering of slices
    The function recognises the two naming conventions that were used in the
    original code.
    """
    patterns = [
        r"^(.+?)_slice_(\d+)",   # bea_tree_slice_000.png , tree_00975_slice_000.png
        r"^(.+?)_(\d{3})\.",     # tree_00638_000.png
    ]
    for pat in patterns:
        m = re.match(pat, filename)
        if m:
            return m.group(1), int(m.group(2))
    # If nothing matches we signal the caller by returning (None, -1)
    return None, -1


def load_image_gray(path: Path) -> np.ndarray:
    """
    Loads a PNG (or any image Pillow can read) as an 8‑bit **grayscale** numpy
    array.  All images are forced to mode “L”.
    """
    img = Image.open(path)
    if img.mode != "L":
        img = img.convert("L")
    return np.array(img, dtype=np.uint8)


# ----------------------------------------------------------------------
# ── Core processing function ───────────────────────────────────────────
# ----------------------------------------------------------------------
def create_multichannel_tiffs(
    input_dir: Path,
    output_dir: Path,
    dataset_name: str,
) -> bool:
    """
    Walks through *input_dir* (expects only *.png files), groups them by the
    extracted tree‑prefix, and writes a 4‑channel TIFF for every slice.

    Returns True on success, False on any fatal error.
    """
    print("\n" + "=" * 60)
    print(f"Processing dataset: {dataset_name}")
    print("=" * 60)

    if not input_dir.is_dir():
        print(f"❌ ERROR: Input folder does not exist → {input_dir}")
        return False

    png_files = sorted(input_dir.glob("*.png"))
    if not png_files:
        print(f"❌ ERROR: No PNG files found in {input_dir}")
        return False

    # --------------------------------------------------------------
    # 1️⃣  Group files by tree prefix (all slices belonging together)
    # --------------------------------------------------------------
    tree_groups: dict[str, dict[int, Path]] = {}
    for f in png_files:
        prefix, slice_idx = get_tree_prefix_and_slice(f.name)
        if prefix is None:
            # Silently ignore files that do not match the expected patterns
            continue
        tree_groups.setdefault(prefix, {})[slice_idx] = f

    if not tree_groups:
        print("❌ ERROR: Could not parse any filenames – check naming convention.")
        return False

    # --------------------------------------------------------------
    # 2️⃣  Create output folder (if needed) and start writing TIFFs
    # --------------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)

    total_written = 0
    for tree_prefix, slices_by_idx in tree_groups.items():
        slice_numbers = sorted(slices_by_idx)
        for cur_idx in tqdm(slice_numbers, desc=f"{tree_prefix}", unit="tiff"):
            # ---- primary channel -------------------------------------------------
            ch0 = load_image_gray(slices_by_idx[cur_idx])

            # ---- next slice (i+1) ------------------------------------------------
            if (cur_idx + 1) in slices_by_idx:
                ch1 = load_image_gray(slices_by_idx[cur_idx + 1])
            else:
                ch1 = np.zeros_like(ch0)          # missing → black

            # ---- slice after next (i+2) -------------------------------------------
            if (cur_idx + 2) in slices_by_idx:
                ch2 = load_image_gray(slices_by_idx[cur_idx + 2])
            else:
                ch2 = np.zeros_like(ch0)          # missing → black

            # ---- index‑gray channel (same value for every pixel) ------------------
            ch3 = np.full_like(ch0, cur_idx, dtype=np.uint8)

            # ---- Stack into a (H, W, 4) array ------------------------------------
            stacked = np.stack([ch0, ch1, ch2, ch3], axis=2)

            # ---- Save as a 4‑channel TIFF ----------------------------------------
            out_path = output_dir / f"{slices_by_idx[cur_idx].stem}.tiff"
            Image.fromarray(stacked, mode="RGBA").save(
                out_path, compression="tiff_deflate"
            )
            total_written += 1

    print(f"\n✅ Done! {total_written} TIFF files written to {output_dir}")
    return True


# ----------------------------------------------------------------------
# ── Entry point ───────────────────────────────────────────────────────
# ----------------------------------------------------------------------
def main():
    """
    Runs the conversion for the *single* dataset `my_yolo_dataset`.
    If you want to point the script at a completely different location,
    simply edit the variables at the top of the file (BASE_DIR, DATA_ROOT,
    DATASET_NAME) or pass the three arguments when calling the script
    from the command line (see the argparse block below).
    """
    # ------------------------------------------------------------------
    # Determine folders (the defaults work for the layout described in the
    # module docstring above)
    # ------------------------------------------------------------------
    input_dir = DATA_ROOT / DATASET_NAME / "images"
    output_dir = DATA_ROOT / DATASET_NAME / "multichannel"

    # ------------------------------------------------------------------
    # Run the conversion
    # ------------------------------------------------------------------
    success = create_multichannel_tiffs(
        input_dir=input_dir,
        output_dir=output_dir,
        dataset_name=DATASET_NAME,
    )
    if not success:
        raise SystemExit("\n❌ Processing failed – see messages above.")

# ----------------------------------------------------------------------
# ── Optional CLI – lets you override the defaults without editing code ─
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # The script works out‑of‑the‑box with the defaults, but for power‑users
    # we expose a tiny CLI that can override them.
    import argparse

    parser = argparse.ArgumentParser(
        description="Create 4‑channel TIFFs for a single YOLO‑11 dataset."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Folder containing PNG slices (default = data/datasets/<name>/images)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Folder where TIFFs will be written (default = .../multichannel)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=DATASET_NAME,
        help="Dataset name used only for logging. Also used to infer default "
             "paths if --input/--output are not supplied.",
    )
    args = parser.parse_args()

    # If the user supplied custom paths use them, otherwise fall back to the
    # defaults defined at the top of the file.
    dataset_name = args.name
    input_dir = Path(args.input) if args.input else DATA_ROOT / dataset_name / "images"
    output_dir = Path(args.output) if args.output else DATA_ROOT / dataset_name / "multichannel"

    # Run the conversion with the possibly‑overridden arguments
    create_multichannel_tiffs(
        input_dir=input_dir,
        output_dir=output_dir,
        dataset_name=dataset_name,
    )