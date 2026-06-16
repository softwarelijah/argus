"""YOLOv8 detector backed by ONNX Runtime.

This is the CPU path that needs neither a GPU nor PyTorch: export a YOLOv8
model to ONNX once (``argus export --weights yolov8n.pt --onnx model.onnx``),
then run the whole detect -> track -> analytics pipeline on real footage with
only numpy, OpenCV and onnxruntime installed.

Shares the :class:`~argus.detection.detector.Detections` contract with the
PyTorch and TensorRT backends, so it is a drop-in replacement.
"""

from __future__ import annotations

import numpy as np

from ..detection.detector import Detections
from ..detection.postprocess import postprocess, preprocess


class ORTDetector:
    """Run a YOLOv8 ONNX model with ONNX Runtime."""

    def __init__(
        self,
        onnx_path: str,
        imgsz: int = 1280,
        conf: float = 0.25,
        iou: float = 0.7,
        num_classes: int = 10,
        names: dict[int, str] | None = None,
        providers: list[str] | None = None,
    ) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - optional dep
            raise ImportError(
                "onnxruntime is required for ORTDetector. "
                "Install with `pip install argus-tracker[export]` or `pip install onnxruntime`."
            ) from exc

        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.num_classes = num_classes
        self.names = names or {}

        providers = providers or ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

        # Honour the model's fixed spatial input size when it is static.
        in_shape = self.session.get_inputs()[0].shape
        if isinstance(in_shape[-1], int) and in_shape[-1] > 0:
            self.imgsz = int(in_shape[-1])

    def __call__(self, frame: np.ndarray) -> Detections:
        return self.detect(frame)

    def detect(self, frame: np.ndarray) -> Detections:
        blob, ratio, pad = preprocess(frame, self.imgsz)
        output = self.session.run(None, {self.input_name: blob})[0]
        boxes, scores, classes = postprocess(
            output,
            ratio,
            pad,
            conf_thresh=self.conf,
            iou_thresh=self.iou,
            num_classes=self.num_classes,
        )
        return Detections(boxes, scores, classes.astype(np.float32), self.names)
