"""
Re-implemented version of the original preprocessing script.

What it does (in order)

1.  Remove the old “gradient_images” and “label_polygon_images” steps.
2.  Write the index-gray image into a folder called ``channel3``.
3.  Move all PNGs from ``obj_train_data`` into a new folder ``images`` and
    rename ``obj_train_data`` → ``annotations``.
4.  Create the folders ``channel1`` and ``channel2``.
5.  Populate channel1:
        * copy every PNG from ``images``,
        * delete the *last* image of each group (same stem, different 3-digit suffix),
        * renumber the remaining files so that the suffix runs from 001-…-N.
6.  Populate channel2:
        * copy everything that is now in ``channel1``,
        * delete the *last* image of each group again,
        * renumber once more (again 001-…-N).
7.  Force the class order in ``obj.names`` to be
        ["twigs", "trunk", "branch", "grass"]
    and rewrite every annotation file in ``annotations`` with the corrected
    class indices.

All operations are performed *in-place* (files are moved/copied, annotation files
are overwritten).  No data is lost except the images that are deliberately
deleted as part of steps 5-6.
"""

from __future__ import annotations

import sys
from pathlib import Path
import shutil
import re
from typing import List, Dict, Tuple

from PIL import Image  # only needed for the index-gray image

# ----------------------------------------------------------------------
# 0. CONFIGURATION
# ----------------------------------------------------------------------
BASE_DIR = Path(
    r"C:\Users\georg\Desktop\a MSc Forestry\Modern Methods in TLS and UAV\Final Project\Script Test"
)

# original locations (kept only for the move-operation)
OBJ_TRAIN_DATA = BASE_DIR / "obj_train_data"
OBJ_NAMES_FILE = BASE_DIR / "obj.names"

# new locations (all created if they do not exist)
IMAGES_DIR = BASE_DIR / "images"
ANNOTATIONS_DIR = BASE_DIR / "annotations"
CHANNEL1_DIR = BASE_DIR / "channel1"
CHANNEL2_DIR = BASE_DIR / "channel2"
CHANNEL3_DIR = BASE_DIR / "channel3"

# Desired order of class names (must be exactly this list)
TARGET_CLASS_ORDER = ["twigs", "trunk", "branch", "grass"]


# ----------------------------------------------------------------------
# 1. HELPERS
# ----------------------------------------------------------------------
def ensure_dir(p: Path) -> None:
    """Create directory if it does not exist."""
    p.mkdir(parents=True, exist_ok=True)


def numeric_suffix(p: Path) -> int:
    """Return the integer that appears after the final '_' and before '.png'."""
    m = re.search(r"_(\d{3})$", p.stem)
    if not m:
        raise ValueError(f"File {p.name} does not match the <name>_### pattern")
    return int(m.group(1))


def stem_without_suffix(p: Path) -> str:
    """Return the part before the final '_'."""
    return re.sub(r"_(\d{3})$", "", p.stem)


def group_by_stem(paths: List[Path]) -> Dict[str, List[Path]]:
    """Group a list of image paths by the common stem (everything before _###)."""
    groups: Dict[str, List[Path]] = {}
    for p in paths:
        stem = stem_without_suffix(p)
        groups.setdefault(stem, []).append(p)
    return groups


def rename_sequentially(paths: List[Path], target_dir: Path) -> None:
    """
    Rename a collection of files that belong to the same stem so that they become
    001, 002, … N (three-digit, zero-padded) inside *target_dir*.
    The original files are assumed to be *paths* (any order); they are moved
    (or copied) into *target_dir* with the new name.
    """
    # sort by the original numeric suffix to keep the temporal order
    paths_sorted = sorted(paths, key=numeric_suffix)
    for new_idx, src in enumerate(paths_sorted, start=1):
        new_name = f"{stem_without_suffix(src)}_{new_idx:03d}.png"
        dst = target_dir / new_name
        shutil.move(str(src), str(dst))


def copy_and_process(source_dir: Path, dest_dir: Path, delete_last: bool) -> None:
    """
    1) copy *all* PNGs from source_dir → dest_dir (preserving the original names);
    2) if *delete_last* is True, remove the highest-numbered file of each group;
    3) renumber the remaining files so they start at 001 and are consecutive.
    """
    ensure_dir(dest_dir)

    # 1) copy everything
    for src in source_dir.glob("*.png"):
        shutil.copy2(src, dest_dir / src.name)

    # 2) group, delete, rename
    groups = group_by_stem(list(dest_dir.glob("*.png")))
    for stem, files in groups.items():
        # sort by numeric suffix so we know which one is the "last"
        files_sorted = sorted(files, key=numeric_suffix)
        if delete_last and len(files_sorted) > 0:
            # delete the highest number
            files_sorted[-1].unlink()
            files = files_sorted[:-1]  # remaining files after deletion
        else:
            files = files_sorted

        # 3) renumber the remaining files
        # we first move them to a temporary location (the same folder) with a
        # temporary name to avoid name clashes during the renaming loop.
        temp_dir = dest_dir / "__tmp_renaming__"
        ensure_dir(temp_dir)

        for f in files:
            f.rename(temp_dir / f.name)

        # now move them back with sequential numbers
        for new_idx, src in enumerate(sorted(temp_dir.iterdir()), start=1):
            new_name = f"{stem}_{new_idx:03d}.png"
            src.rename(dest_dir / new_name)

        # clean up temp folder
        shutil.rmtree(temp_dir)


def read_class_names(obj_names_file: Path) -> List[str]:
    """Read obj.names, strip whitespace and ignore empty lines."""
    with obj_names_file.open("r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def rewrite_annotation_file(
    anno_file: Path,
    old_to_new: Dict[int, int],
) -> None:
    """
    Read a YOLO .txt file, replace the first integer (class index) according to
    *old_to_new* and write the file back in place.
    """
    lines_out = []
    with anno_file.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            old_idx = int(parts[0])
            new_idx = old_to_new.get(old_idx, old_idx)  # fallback to itself
            parts[0] = str(new_idx)
            lines_out.append(" ".join(parts))

    with anno_file.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines_out) + ("\n" if lines_out else ""))


# ----------------------------------------------------------------------
# 2. MAIN PIPELINE
# ----------------------------------------------------------------------
def main() -> None:
    # ------------------------------------------------------------------
    # (a)  Ensure all needed directories exist
    # ------------------------------------------------------------------
    ensure_dir(IMAGES_DIR)
    ensure_dir(ANNOTATIONS_DIR)
    ensure_dir(CHANNEL1_DIR)
    ensure_dir(CHANNEL2_DIR)
    ensure_dir(CHANNEL3_DIR)

    # ------------------------------------------------------------------
    # (b)  Move PNGs from obj_train_data → images
    # ------------------------------------------------------------------
    pngs = list(OBJ_TRAIN_DATA.glob("*.png"))
    if not pngs:
        print("WARNING:  No PNG files found in obj_train_data - nothing to move.")
    else:
        for p in pngs:
            shutil.move(str(p), IMAGES_DIR / p.name)

    # (c)  Rename the folder obj_train_data → annotations
    if OBJ_TRAIN_DATA.exists():
        if ANNOTATIONS_DIR.exists():
            # should not happen, but we merge the old content just in case
            for p in OBJ_TRAIN_DATA.iterdir():
                shutil.move(str(p), ANNOTATIONS_DIR / p.name)
            OBJ_TRAIN_DATA.rmdir()
        else:
            OBJ_TRAIN_DATA.rename(ANNOTATIONS_DIR)

    # ------------------------------------------------------------------
    # (d)  Create the index-gray image (channel3)
    # ------------------------------------------------------------------
    # we keep the original logic for the gray value but now write to channel3
    image_paths = sorted(IMAGES_DIR.glob("*.png"))
    n_images = len(image_paths)

    if n_images == 0:
        print("ERROR: No images found in the newly created 'images' folder - abort.")
        sys.exit(1)

    print(f"INFO:  {n_images} PNG images will be processed (index-gray into channel3).")

    for i, img_path in enumerate(image_paths):
        img = Image.open(img_path)
        w, h = img.size

        # gray value is linear from 0 … 255 over the whole dataset
        if n_images == 1:
            gray_val = 0
        else:
            gray_val = int(round((i / (n_images - 1)) * 255))

        gray_arr = (gray_val * (np.ones((h, w), dtype=np.uint8))).astype(np.uint8)
        # Pillow can write directly from a numpy array
        Image.fromarray(gray_arr).save(CHANNEL3_DIR / img_path.name)

    # ------------------------------------------------------------------
    # (e)  Populate channel1 (delete last of each group, renumber)
    # ------------------------------------------------------------------
    copy_and_process(
        source_dir=IMAGES_DIR,
        dest_dir=CHANNEL1_DIR,
        delete_last=True,  # “delete the last image of each unique name”
    )
    print("SUCCESS: channel1 populated (first deletion & renumber).")

    # ------------------------------------------------------------------
    # (f)  Populate channel2 (copy from channel1, delete last again, renumber)
    # ------------------------------------------------------------------
    copy_and_process(
        source_dir=CHANNEL1_DIR,
        dest_dir=CHANNEL2_DIR,
        delete_last=True,  # second deletion step
    )
    print("SUCCESS: channel2 populated (second deletion & renumber).")

    # ------------------------------------------------------------------
    # (g)  Fix obj.names order and rewrite all annotation files
    # ------------------------------------------------------------------
    current_names = read_class_names(OBJ_NAMES_FILE)
    print(f"INFO: Current obj.names ({len(current_names)} lines): {current_names}")

    # Build a map: old index → new index (according to TARGET_CLASS_ORDER)
    # If the file already matches the desired order we skip everything.
    if current_names == TARGET_CLASS_ORDER:
        print("SUCCESS: obj.names already in the desired order - no changes needed.")
    else:
        # Build name → old index mapping
        name_to_old_idx = {name: idx for idx, name in enumerate(current_names)}
        # Build old → new index mapping (only for names that exist in both lists)
        old_to_new: Dict[int, int] = {}
        for new_idx, name in enumerate(TARGET_CLASS_ORDER):
            old_idx = name_to_old_idx.get(name)
            if old_idx is None:
                raise RuntimeError(
                    f"Required class name '{name}' is missing from obj.names."
                )
            old_to_new[old_idx] = new_idx

        # 1) rewrite every annotation file in the *annotations* folder
        anno_files = list(ANNOTATIONS_DIR.glob("*.txt"))
        for txt in anno_files:
            rewrite_annotation_file(txt, old_to_new)

        # 2) overwrite obj.names with the correct order
        with OBJ_NAMES_FILE.open("w", encoding="utf-8") as f:
            for name in TARGET_CLASS_ORDER:
                f.write(name + "\n")

        print("INFO: obj.names reordered and all annotation files updated.")

    print("\nSUCCESS: All steps completed successfully.")


# ----------------------------------------------------------------------
# 3. ENTRY POINT
# ----------------------------------------------------------------------
if __name__ == "__main__":
    try:
        # numpy is used only for the index-gray creation (quick & easy)
        import numpy as np
    except ImportError:
        sys.stderr.write("ERROR: This script needs numpy (for the index-gray image).\n")
        sys.exit(1)

    try:
        main()
    except Exception as exc:  # pragma: no cover
        sys.stderr.write(f"ERROR: Fatal error: {exc}\n")
        sys.exit(1)
