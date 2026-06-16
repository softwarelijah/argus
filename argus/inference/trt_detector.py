"""Detector backed by a TensorRT engine.

Shares the :class:`~argus.detection.detector.Detections` output contract with
:class:`~argus.detection.detector.YOLODetector`, so the pipeline and tracker do
not care which backend produced the boxes.
"""

from __future__ import annotations

import numpy as np

from ..detection.detector import Detections
from ..detection.postprocess import postprocess, preprocess
from .tensorrt_engine import TRTEngine


class TRTDetector:
    """YOLOv8 detector running on a serialized TensorRT engine."""

    def __init__(
        self,
        engine_path: str,
        imgsz: int = 1280,
        conf: float = 0.25,
        iou: float = 0.7,
        num_classes: int = 10,
        names: dict[int, str] | None = None,
    ) -> None:
        self.engine = TRTEngine(engine_path)
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.num_classes = num_classes
        self.names = names or {}

    def __call__(self, frame: np.ndarray) -> Detections:
        return self.detect(frame)

    def detect(self, frame: np.ndarray) -> Detections:
        blob, ratio, pad = preprocess(frame, self.imgsz)
        output = self.engine.infer(blob)[0]
        boxes, scores, classes = postprocess(
            output,
            ratio,
            pad,
            conf_thresh=self.conf,
            iou_thresh=self.iou,
            num_classes=self.num_classes,
        )
        return Detections(boxes, scores, classes.astype(np.float32), self.names)
