import numpy as np

from argus.detection.postprocess import letterbox, nms, postprocess


def test_letterbox_output_shape_and_ratio():
    img = np.zeros((720, 1280, 3), dtype=np.uint8)
    padded, ratio, (dw, dh) = letterbox(img, (640, 640))
    assert padded.shape == (640, 640, 3)
    assert ratio == 640 / 1280  # limited by the wider dimension
    assert dh > 0 and np.isclose(dw, 0, atol=1.0)


def test_nms_removes_overlapping_boxes():
    boxes = np.array(
        [[0, 0, 10, 10], [1, 1, 11, 11], [100, 100, 110, 110]], dtype=np.float32
    )
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    keep = nms(boxes, scores, iou_thresh=0.5)
    assert 0 in keep
    assert 2 in keep
    assert 1 not in keep  # suppressed by box 0


def test_nms_empty():
    assert nms(np.empty((0, 4), dtype=np.float32), np.empty((0,), np.float32), 0.5) == []


def test_postprocess_decodes_single_box():
    # Craft a YOLOv8-style output (1, 4+nc, anchors) with one strong anchor.
    num_classes = 10
    anchors = 3
    out = np.zeros((1, 4 + num_classes, anchors), dtype=np.float32)
    # anchor 0: centre (320, 320), w/h 40, class 3 confident
    out[0, 0, 0] = 320
    out[0, 1, 0] = 320
    out[0, 2, 0] = 40
    out[0, 3, 0] = 40
    out[0, 4 + 3, 0] = 0.9

    boxes, scores, classes = postprocess(
        out, ratio=1.0, pad=(0.0, 0.0), conf_thresh=0.25, num_classes=num_classes
    )
    assert len(boxes) == 1
    assert classes[0] == 3
    assert np.isclose(scores[0], 0.9, atol=1e-5)
    # decoded xyxy around the centre
    assert np.allclose(boxes[0], [300, 300, 340, 340], atol=1e-4)


def test_postprocess_respects_confidence_threshold():
    num_classes = 10
    out = np.zeros((1, 4 + num_classes, 2), dtype=np.float32)
    out[0, 4, 0] = 0.1  # below threshold
    boxes, scores, classes = postprocess(out, 1.0, (0.0, 0.0), conf_thresh=0.25)
    assert len(boxes) == 0
