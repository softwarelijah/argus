"""End-to-end test of the ONNX Runtime backend on CPU.

Builds a minimal ONNX graph with the YOLOv8 output signature
``(1, 4 + num_classes, anchors)`` and runs a real frame through ORTDetector,
proving the preprocess -> onnxruntime -> postprocess path works without a GPU
or PyTorch.
"""

from __future__ import annotations

import numpy as np
import pytest

ort = pytest.importorskip("onnxruntime")
onnx = pytest.importorskip("onnx")

from argus.inference.onnx_detector import ORTDetector  # noqa: E402


def _build_dummy_yolo_onnx(path, imgsz=64, num_classes=10):
    """A graph that emits one fixed detection regardless of input."""
    from onnx import TensorProto, helper

    anchors = 2
    out = np.zeros((1, 4 + num_classes, anchors), dtype=np.float32)
    # anchor 0: centre (32, 32), size 10x10, class 3 at score 0.9
    out[0, 0, 0], out[0, 1, 0] = 32.0, 32.0
    out[0, 2, 0], out[0, 3, 0] = 10.0, 10.0
    out[0, 4 + 3, 0] = 0.9

    const = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["output0"],
        value=helper.make_tensor(
            "v", TensorProto.FLOAT, out.shape, out.flatten().tolist()
        ),
    )
    inp = helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, imgsz, imgsz])
    outp = helper.make_tensor_value_info("output0", TensorProto.FLOAT, list(out.shape))
    graph = helper.make_graph([const], "dummy_yolo", [inp], [outp])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 9
    onnx.save(model, str(path))


def test_ort_detector_decodes_detection(tmp_path):
    model_path = tmp_path / "dummy.onnx"
    _build_dummy_yolo_onnx(model_path, imgsz=64, num_classes=10)

    detector = ORTDetector(str(model_path), num_classes=10, conf=0.25, names={3: "car"})
    assert detector.imgsz == 64  # picked up from the model input shape

    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    det = detector.detect(frame)

    assert len(det) == 1
    assert int(det.classes[0]) == 3
    assert det.scores[0] == pytest.approx(0.9, abs=1e-5)
    # box centred at (32, 32) with size 10 -> roughly [27, 27, 37, 37]
    assert det.boxes[0] == pytest.approx([27, 27, 37, 37], abs=1.0)


def test_ort_detector_feeds_tracker(tmp_path):
    from argus.pipeline import VideoPipeline
    from argus.tracking import TrackerConfig

    model_path = tmp_path / "dummy.onnx"
    _build_dummy_yolo_onnx(model_path, imgsz=64, num_classes=10)
    detector = ORTDetector(str(model_path), num_classes=10, names={3: "car"})

    pipe = VideoPipeline(detector, TrackerConfig(frame_rate=30, new_track_thresh=0.5), draw=False)
    tracks_seen = 0
    for i in range(1, 6):
        result = pipe.process_frame(np.zeros((64, 64, 3), np.uint8), i)
        tracks_seen = max(tracks_seen, len(result.tracks))
    assert tracks_seen == 1
