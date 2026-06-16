"""Slicing Aided Hyper Inference (SAHI) for small-object detection.

VisDrone targets are tiny relative to the frame, so a single downscaled forward
pass loses them. SAHI runs the detector over overlapping tiles at native scale,
then maps the boxes back and merges across tiles with NMS. It trades latency for
a large recall gain on small objects, which is exactly the aerial regime.

The base detector is duck-typed: any object with ``detect(frame) -> Detections``
works, so this wraps both the PyTorch and TensorRT backends.
"""

from __future__ import annotations

import numpy as np

from .detector import Detections
from .postprocess import nms


def generate_slices(
    height: int,
    width: int,
    slice_h: int = 640,
    slice_w: int = 640,
    overlap: float = 0.2,
) -> list[tuple[int, int, int, int]]:
    """Return (x1, y1, x2, y2) tile boxes covering an image with overlap."""
    step_h = max(1, int(slice_h * (1 - overlap)))
    step_w = max(1, int(slice_w * (1 - overlap)))

    slices: list[tuple[int, int, int, int]] = []
    y = 0
    while y < height:
        y2 = min(y + slice_h, height)
        y1 = max(0, y2 - slice_h)
        x = 0
        while x < width:
            x2 = min(x + slice_w, width)
            x1 = max(0, x2 - slice_w)
            slices.append((x1, y1, x2, y2))
            if x2 >= width:
                break
            x += step_w
        if y2 >= height:
            break
        y += step_h
    # De-duplicate edge-clamped tiles.
    return list(dict.fromkeys(slices))


class SlicedDetector:
    """Wrap a base detector with SAHI tiled inference."""

    def __init__(
        self,
        detector,
        slice_h: int = 640,
        slice_w: int = 640,
        overlap: float = 0.2,
        iou_thresh: float = 0.6,
        include_full_frame: bool = True,
    ) -> None:
        self.detector = detector
        self.slice_h = slice_h
        self.slice_w = slice_w
        self.overlap = overlap
        self.iou_thresh = iou_thresh
        self.include_full_frame = include_full_frame
        self.names = getattr(detector, "names", {})

    def __call__(self, frame: np.ndarray) -> Detections:
        return self.detect(frame)

    def detect(self, frame: np.ndarray) -> Detections:
        h, w = frame.shape[:2]
        all_boxes: list[np.ndarray] = []
        all_scores: list[np.ndarray] = []
        all_classes: list[np.ndarray] = []

        slices = generate_slices(h, w, self.slice_h, self.slice_w, self.overlap)
        for x1, y1, x2, y2 in slices:
            tile = frame[y1:y2, x1:x2]
            det = self.detector.detect(tile)
            if len(det) == 0:
                continue
            boxes = det.boxes.copy()
            boxes[:, [0, 2]] += x1  # shift back to full-frame coordinates
            boxes[:, [1, 3]] += y1
            all_boxes.append(boxes)
            all_scores.append(det.scores)
            all_classes.append(det.classes)

        # A full-frame pass catches large objects a small tile would clip.
        if self.include_full_frame:
            det = self.detector.detect(frame)
            if len(det) > 0:
                all_boxes.append(det.boxes)
                all_scores.append(det.scores)
                all_classes.append(det.classes)

        if not all_boxes:
            return Detections(
                np.empty((0, 4), np.float32),
                np.empty((0,), np.float32),
                np.empty((0,), np.float32),
                self.names,
            )

        boxes = np.concatenate(all_boxes, axis=0)
        scores = np.concatenate(all_scores, axis=0)
        classes = np.concatenate(all_classes, axis=0)

        # Class-aware NMS to merge duplicate detections from overlapping tiles.
        keep_all: list[int] = []
        for cls_id in np.unique(classes):
            idx = np.where(classes == cls_id)[0]
            keep = nms(boxes[idx], scores[idx], self.iou_thresh)
            keep_all.extend(idx[keep].tolist())
        keep_idx = np.asarray(sorted(keep_all), dtype=int)

        return Detections(
            boxes[keep_idx].astype(np.float32),
            scores[keep_idx].astype(np.float32),
            classes[keep_idx].astype(np.float32),
            self.names,
        )
