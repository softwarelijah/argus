"""Build an INT8 calibration tensor from representative VisDrone frames.

TensorRT INT8 entropy calibration estimates per-tensor activation ranges from a
small, representative sample. For aerial deployment the sample must come from
the target distribution (drone imagery), so we draw frames from the VisDrone
images rather than COCO.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..detection.postprocess import preprocess


def build_calibration_set(
    images_dir: str | Path,
    num_samples: int = 256,
    imgsz: int = 1280,
    stride: int | None = None,
) -> np.ndarray:
    """Return an ``(N, 3, imgsz, imgsz)`` float32 array of preprocessed frames.

    Frames are sampled evenly across the directory so the calibration set spans
    varied scenes rather than a single contiguous clip.
    """
    import cv2

    images_dir = Path(images_dir)
    paths = sorted(images_dir.glob("*.jpg"))
    if not paths:
        raise FileNotFoundError(f"no .jpg images found in {images_dir}")

    if stride is None:
        stride = max(1, len(paths) // num_samples)
    selected = paths[::stride][:num_samples]

    blobs = []
    for path in selected:
        img = cv2.imread(str(path))
        if img is None:
            continue
        blob, _, _ = preprocess(img, imgsz)
        blobs.append(blob[0])  # drop the batch dim added by preprocess
    if not blobs:
        raise RuntimeError("failed to read any calibration images")
    return np.ascontiguousarray(np.stack(blobs), dtype=np.float32)
