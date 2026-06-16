"""VisDrone dataset utilities: class map and YOLO-format conversion.

VisDrone-DET annotations are one CSV line per object:

    bbox_left, bbox_top, bbox_width, bbox_height, score, category,
    truncation, occlusion

Categories 0 (ignored regions) and 11 (others) are dropped for training,
leaving the 10 evaluation classes remapped to contiguous ids 0..9.
"""

from __future__ import annotations

from pathlib import Path

# Original VisDrone category id -> training class name. Ids 0 and 11 are
# excluded from training and evaluation.
VISDRONE_RAW_CLASSES = {
    0: "ignored",
    1: "pedestrian",
    2: "people",
    3: "bicycle",
    4: "car",
    5: "van",
    6: "truck",
    7: "tricycle",
    8: "awning-tricycle",
    9: "bus",
    10: "motor",
    11: "others",
}

# Contiguous 0..9 class names used by the model.
VISDRONE_CLASSES = [
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
]

# Map raw category (1..10) -> contiguous training id (0..9).
_RAW_TO_TRAIN = {raw: raw - 1 for raw in range(1, 11)}


def names_dict() -> dict[int, str]:
    """Return the {id: name} map the detector and visualizer expect."""
    return dict(enumerate(VISDRONE_CLASSES))


def _convert_annotation(
    ann_path: Path, img_w: int, img_h: int
) -> list[str]:
    """Convert one VisDrone annotation file to YOLO label lines."""
    lines: list[str] = []
    for raw in ann_path.read_text().splitlines():
        raw = raw.strip().rstrip(",")
        if not raw:
            continue
        parts = raw.split(",")
        if len(parts) < 6:
            continue
        x, y, w, h = (float(p) for p in parts[:4])
        category = int(parts[5])
        if category not in _RAW_TO_TRAIN:
            continue  # skip ignored regions and 'others'
        if w <= 0 or h <= 0:
            continue

        cls = _RAW_TO_TRAIN[category]
        cx = (x + w / 2) / img_w
        cy = (y + h / 2) / img_h
        nw = w / img_w
        nh = h / img_h
        # Clamp to the valid normalised range.
        cx, cy = min(max(cx, 0.0), 1.0), min(max(cy, 0.0), 1.0)
        nw, nh = min(max(nw, 0.0), 1.0), min(max(nh, 0.0), 1.0)
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    return lines


def convert_split(root: str | Path, split_dir: str | Path) -> int:
    """Convert one VisDrone split (e.g. ``VisDrone2019-DET-train``) to YOLO.

    Reads ``images/`` and ``annotations/`` under ``split_dir`` and writes YOLO
    ``.txt`` files into a sibling ``labels/`` directory. Returns the number of
    images converted. Requires OpenCV to read image dimensions.
    """
    import cv2

    split_dir = Path(split_dir)
    images_dir = split_dir / "images"
    ann_dir = split_dir / "annotations"
    labels_dir = split_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for img_path in sorted(images_dir.glob("*.jpg")):
        ann_path = ann_dir / f"{img_path.stem}.txt"
        if not ann_path.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        lines = _convert_annotation(ann_path, w, h)
        (labels_dir / f"{img_path.stem}.txt").write_text("\n".join(lines))
        count += 1
    return count


def write_data_yaml(root: str | Path, out_path: str | Path) -> Path:
    """Write the Ultralytics dataset YAML for VisDrone."""
    import yaml

    root = Path(root).resolve()
    data = {
        "path": str(root),
        "train": "VisDrone2019-DET-train/images",
        "val": "VisDrone2019-DET-val/images",
        "test": "VisDrone2019-DET-test-dev/images",
        "nc": len(VISDRONE_CLASSES),
        "names": VISDRONE_CLASSES,
    }
    out_path = Path(out_path)
    out_path.write_text(yaml.safe_dump(data, sort_keys=False))
    return out_path
