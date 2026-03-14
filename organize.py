from pathlib import Path
import shutil
import re
import numpy as np
from PIL import Image


# ----------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------

BASE_DIR = Path(
    r"C:\Users\georg\Desktop\a MSc Forestry\Modern Methods in TLS and UAV\Final Project\test data"
)

CHANNEL0_DIR = BASE_DIR / "channel0"
CHANNEL1_DIR = BASE_DIR / "channel1"
CHANNEL2_DIR = BASE_DIR / "channel2"
CHANNEL3_DIR = BASE_DIR / "channel3"


# ----------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------

def ensure_dir(folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)


def clear_pngs(folder: Path) -> None:
    if folder.exists():
        for f in folder.glob("*.png"):
            f.unlink()


def numeric_suffix(p: Path) -> int:
    m = re.search(r"_(\d+)$", p.stem)
    if not m:
        raise ValueError(f"Filename does not match *_###.png pattern: {p.name}")
    return int(m.group(1))


def stem_without_suffix(p: Path) -> str:
    return re.sub(r"_(\d+)$", "", p.stem)


def group_by_stem(paths):
    groups = {}
    for p in paths:
        stem = stem_without_suffix(p)
        groups.setdefault(stem, []).append(p)
    return groups


# ----------------------------------------------------------------------
# CORE CHANNEL BUILDER
# ----------------------------------------------------------------------

def build_channel(source_dir: Path, dest_dir: Path):
    ensure_dir(dest_dir)
    clear_pngs(dest_dir)

    # Copy silently
    for src in sorted(source_dir.glob("*.png")):
        shutil.copy2(src, dest_dir / src.name)

    groups = group_by_stem(list(dest_dir.glob("*.png")))

    for stem, files in groups.items():

        files_sorted = sorted(files, key=numeric_suffix)

        # ✅ Delete last file
        if files_sorted:
            to_delete = files_sorted[-1]
            print(f"DELETE: {dest_dir.name}/{to_delete.name}")
            to_delete.unlink()
            files_sorted = files_sorted[:-1]

        # ✅ Shift all remaining indices +1
        # Rename from highest to lowest to avoid collisions
        for file in sorted(files_sorted, key=numeric_suffix, reverse=True):
            old_index = numeric_suffix(file)
            new_index = old_index + 1
            new_name = f"{stem}_{new_index:03d}.png"

            print(f"RENAME: {dest_dir.name}/{file.name} → {new_name}")
            file.rename(dest_dir / new_name)


# ----------------------------------------------------------------------
# CHANNEL3 GRADIENT
# ----------------------------------------------------------------------

def build_channel3_gradient(source_dir: Path, dest_dir: Path):
    ensure_dir(dest_dir)
    clear_pngs(dest_dir)

    image_paths = sorted(source_dir.glob("*.png"))
    n_images = len(image_paths)

    if n_images == 0:
        return

    for i, img_path in enumerate(image_paths):
        img = Image.open(img_path)
        w, h = img.size

        gray_val = 0 if n_images == 1 else int(round((i / (n_images - 1)) * 255))
        gray_arr = np.full((h, w), gray_val, dtype=np.uint8)

        Image.fromarray(gray_arr).save(dest_dir / img_path.name)


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------

def main():

    if not CHANNEL0_DIR.exists():
        print("channel0 does not exist. Aborting.")
        return

    ensure_dir(CHANNEL1_DIR)
    ensure_dir(CHANNEL2_DIR)
    ensure_dir(CHANNEL3_DIR)

    # channel0 → channel1
    build_channel(CHANNEL0_DIR, CHANNEL1_DIR)

    # channel1 → channel2
    build_channel(CHANNEL1_DIR, CHANNEL2_DIR)

    # channel0 → channel3
    build_channel3_gradient(CHANNEL0_DIR, CHANNEL3_DIR)

    print("\n✅ DONE")


if __name__ == "__main__":
    main()