"""The config-driven detector builder wires backends from YAML."""

import numpy as np
import pytest

pytest.importorskip("onnxruntime")
pytest.importorskip("onnx")

from argus.cli import _detector_from_config  # noqa: E402
from argus.detection.sahi import SlicedDetector  # noqa: E402
from argus.inference.onnx_detector import ORTDetector  # noqa: E402
from argus.utils.config import Config  # noqa: E402
from tests.test_onnx_detector import _build_dummy_yolo_onnx  # noqa: E402


def test_build_onnx_detector_from_config(tmp_path):
    model = tmp_path / "m.onnx"
    _build_dummy_yolo_onnx(model, imgsz=64, num_classes=10)
    cfg = Config({"detector": {"backend": "onnx", "onnx": str(model), "imgsz": 64}})
    det = _detector_from_config(cfg)
    assert isinstance(det, ORTDetector)
    out = det.detect(np.zeros((64, 64, 3), np.uint8))
    assert len(out) == 1


def test_sahi_wrapping_from_config(tmp_path):
    model = tmp_path / "m.onnx"
    _build_dummy_yolo_onnx(model, imgsz=64, num_classes=10)
    cfg = Config(
        {
            "detector": {"backend": "onnx", "onnx": str(model), "imgsz": 64},
            "sahi": {"enabled": True, "slice": 64, "overlap": 0.2},
        }
    )
    det = _detector_from_config(cfg)
    assert isinstance(det, SlicedDetector)
