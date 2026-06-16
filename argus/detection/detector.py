"""YOLOv8 detector wrapper producing tracker-ready detections."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Detections:
    """A batch of detections for a single frame.

    Boxes are ``(N, 4)`` in (x1, y1, x2, y2) pixel coordinates. ``scores`` and
    ``classes`` are length ``N``.
    """

    boxes: np.ndarray
    scores: np.ndarray
    classes: np.ndarray
    names: dict[int, str] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.boxes)

    def to_array(self) -> np.ndarray:
        """Pack into the ``(N, 6)`` ``[x1, y1, x2, y2, score, cls]`` layout the
        tracker consumes."""
        if len(self) == 0:
            return np.empty((0, 6), dtype=np.float32)
        return np.concatenate(
            [
                self.boxes.astype(np.float32),
                self.scores.reshape(-1, 1).astype(np.float32),
                self.classes.reshape(-1, 1).astype(np.float32),
            ],
            axis=1,
        )

    def filter(self, min_score: float) -> Detections:
        keep = self.scores >= min_score
        return Detections(self.boxes[keep], self.scores[keep], self.classes[keep], self.names)


class YOLODetector:
    """Thin wrapper around an Ultralytics YOLOv8 model.

    The heavy ``ultralytics`` import is deferred until construction so the rest
    of the library (tracking, data tools) can be imported on machines without a
    full deep learning stack installed.
    """

    def __init__(
        self,
        weights: str = "yolov8s.pt",
        conf: float = 0.25,
        iou: float = 0.7,
        imgsz: int = 1280,
        device: str | None = None,
        half: bool = True,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - depends on optional dep
            raise ImportError(
                "ultralytics is required for YOLODetector. Install with "
                "`pip install argus-tracker[train]`."
            ) from exc

        self.model = YOLO(weights)
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.device = device
        self.half = half
        self.names = dict(self.model.names) if hasattr(self.model, "names") else {}

    def __call__(self, frame: np.ndarray) -> Detections:
        return self.detect(frame)

    def detect(self, frame: np.ndarray) -> Detections:
        """Run detection on a single BGR frame (as read by OpenCV)."""
        result = self.model.predict(
            frame,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            half=self.half,
            verbose=False,
        )[0]

        if result.boxes is None or len(result.boxes) == 0:
            return Detections(
                np.empty((0, 4), np.float32),
                np.empty((0,), np.float32),
                np.empty((0,), np.float32),
                self.names,
            )

        boxes = result.boxes.xyxy.cpu().numpy().astype(np.float32)
        scores = result.boxes.conf.cpu().numpy().astype(np.float32)
        classes = result.boxes.cls.cpu().numpy().astype(np.float32)
        return Detections(boxes, scores, classes, self.names)
