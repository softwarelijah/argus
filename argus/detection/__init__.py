"""Detection subpackage: YOLOv8 detector and raw-output post-processing."""

from .detector import Detections, YOLODetector
from .postprocess import letterbox, nms, postprocess, preprocess
from .sahi import SlicedDetector, generate_slices

__all__ = [
    "Detections",
    "YOLODetector",
    "SlicedDetector",
    "generate_slices",
    "letterbox",
    "preprocess",
    "postprocess",
    "nms",
]
