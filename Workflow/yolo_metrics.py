from __future__ import annotations
import shutil
from pathlib import Path
from ultralytics import YOLO
import warnings

warnings.filterwarnings("ignore")

IMAGE_EXTENSIONS = {".tif", ".png"}

MODEL_PATH = Path("Model") / "best.pt"
IMAGES_DIR = Path("Images")
TEST_DIR = Path("TEST")
OUTPUT_ROOT = Path("OUTPUT") / "yolo_metrics"
TREE_FILTER: list[str] | None = None
CONFIDENCE = 0.10
IOU_THRESHOLD = 0.70
IMAGE_SIZE = 640
DEVICE = "cpu"
ENABLE_PLOTS = True
DATASET_NAME = "tmp_dataset_all"

# Load model & class names
model = YOLO(str(MODEL_PATH))
names_attr = getattr(model, "names", None)
class_names = (
    [names_attr[i] for i in sorted(names_attr.keys())]
    if isinstance(names_attr, dict)
    else [str(x) for x in names_attr]
)

# Prepare output dirs
dataset_root = OUTPUT_ROOT / DATASET_NAME
shutil.rmtree(dataset_root, ignore_errors=True)

images_out = dataset_root / "images" / "test"
labels_out = dataset_root / "labels" / "test"
images_out.mkdir(parents=True, exist_ok=True)
labels_out.mkdir(parents=True, exist_ok=True)

# Build image stem → path index
image_index: dict[str, Path] = {
    p.stem: p
    for p in sorted(IMAGES_DIR.iterdir())
    if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
}

# Collect valid tree dirs
tree_dirs = [
    item
    for item in sorted(TEST_DIR.iterdir())
    if item.is_dir()
    and (item / "obj.names").exists()
    and (item / "obj_train_data").exists()
]

# Process each tree dir
copied_images = 0
written_labels = 0
skipped_missing_image = 0

for tree_dir in tree_dirs:
    ref_names = [
        line.strip()
        for line in (tree_dir / "obj.names").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    # No remapping: use reference file class IDs directly
    class_mapping = {ref_id: ref_id for ref_id in range(len(ref_names))}

    for gt_file in sorted((tree_dir / "obj_train_data").glob("*.txt")):
        stem_candidates = [gt_file.stem, f"tree_{gt_file.stem}"] + (
            [gt_file.stem[5:]] if gt_file.stem.startswith("tree_") else []
        )
        image_path = next(
            (image_index[c] for c in stem_candidates if c in image_index), None
        )
        if image_path is None:
            skipped_missing_image += 1
            continue

        raw_lines = [
            line.strip()
            for line in gt_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        remapped_lines = [
            " ".join([str(class_mapping.get(int(float(parts[0])), int(float(parts[0])))) ] + parts[1:])
            for parts in (line.split() for line in raw_lines)
            if len(parts) >= 5
            and int(float(parts[0])) >= 0  # valid parse guard
        ]

        out_stem = "".join(
            ch if ch.isalnum() or ch in {"-", "_"} else "_"
            for ch in f"{tree_dir.name}__{image_path.stem}"
        ) if remapped_lines else ""

        if remapped_lines:
            out_image = images_out / f"{out_stem}{image_path.suffix.lower()}"
            if not out_image.exists():
                shutil.copy2(image_path, out_image)
                copied_images += 1

        _ = (labels_out / f"{out_stem}.txt").write_text(
            "\n".join(remapped_lines) + "\n", encoding="utf-8"
        ) if remapped_lines else None

        written_labels += bool(remapped_lines)

# Write data.yaml
data_yaml = dataset_root / "data.yaml"
data_yaml.write_text(
    "\n".join([
        f"path: {dataset_root.as_posix()}",
        "train: images/test",
        "val: images/test",
        "test: images/test",
        f"nc: {len(class_names)}",
        "names:",
        *[f'  {i}: "{name.replace(chr(34), chr(92)+chr(34))}"' for i, name in enumerate(class_names)],
    ]) + "\n",
    encoding="utf-8",
)

# Validate
print(f"Prepared dataset | images: {copied_images} | labels: {written_labels} | skipped: {skipped_missing_image}")

results = model.val(
    data=str(data_yaml),
    imgsz=IMAGE_SIZE,
    conf=CONFIDENCE,
    iou=IOU_THRESHOLD,
    plots=ENABLE_PLOTS,
    device=DEVICE,
    verbose=False,
)

# Print summary
box = getattr(results, "box", None)
mp = getattr(box, "mp", None)
mr = getattr(box, "mr", None)
f1 = 2 * mp * mr / (mp + mr)

print(f"map50:          {getattr(box, 'map50', None):.4f}")
print(f"map50-95:       {getattr(box, 'map', None):.4f}")
print(f"map75:          {getattr(box, 'map75', None):.4f}")
print(f"mean precision: {mp:.4f}")
print(f"mean recall:    {mr:.4f}")
print(f"f1 score:       {f1:.4f}")
