#!/usr/bin/env python3
"""
Combine 4 grayscale PNG folders (channel0–channel3) into RGBA TIFF files.
"""

from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------
# Fixed folder paths (Windows)
# ---------------------------------------------------------------------
BASE_DIR = Path(
    r"C:\Users\georg\Desktop\a MSc Forestry\Modern Methods in TLS and UAV\Final Project\test data"
)

CHANNEL0_DIR = BASE_DIR / "channel0"   # R
CHANNEL1_DIR = BASE_DIR / "channel1"   # G
CHANNEL2_DIR = BASE_DIR / "channel2"   # B
CHANNEL3_DIR = BASE_DIR / "channel3"   # A

OUTPUT_DIR = BASE_DIR / "multichannel_tifs"


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------
def load_gray(path: Path) -> np.ndarray:
    """Load image as 8-bit grayscale numpy array."""
    img = Image.open(path)
    if img.mode != "L":
        img = img.convert("L")
    return np.array(img, dtype=np.uint8)


def get_matching_file(stem: str, folder: Path) -> Path | None:
    """Return matching PNG file from folder (same filename)."""
    candidate = folder / f"{stem}.png"
    return candidate if candidate.exists() else None


# ---------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------
def main():

    if not CHANNEL0_DIR.exists():
        raise FileNotFoundError(f"Channel0 folder not found: {CHANNEL0_DIR}")

    png_files = sorted(CHANNEL0_DIR.glob("*.png"))
    if not png_files:
        raise FileNotFoundError("No PNG files found in channel0 folder.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Writing TIFFs to:\n{OUTPUT_DIR}\n")

    for r_path in tqdm(png_files, desc="Processing", unit="image"):

        stem = r_path.stem

        # Load R channel
        ch_r = load_gray(r_path)

        # Load other channels (or zeros if missing)
        g_path = get_matching_file(stem, CHANNEL1_DIR)
        b_path = get_matching_file(stem, CHANNEL2_DIR)
        a_path = get_matching_file(stem, CHANNEL3_DIR)

        ch_g = load_gray(g_path) if g_path else np.zeros_like(ch_r)
        ch_b = load_gray(b_path) if b_path else np.zeros_like(ch_r)
        ch_a = load_gray(a_path) if a_path else np.zeros_like(ch_r)

        # Stack channels into RGBA
        stacked = np.stack([ch_r, ch_g, ch_b, ch_a], axis=2)

        # Save as TIFF
        out_path = OUTPUT_DIR / f"{stem}.tif"
        Image.fromarray(stacked, mode="RGBA").save(
            out_path,
            compression="tiff_deflate"
        )

    print("\n✅ Finished creating multichannel TIFFs.")


# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()