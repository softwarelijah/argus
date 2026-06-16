import numpy as np

from argus.tracking.matching import fuse_score, iou_distance, ious, linear_assignment


def test_iou_identical_boxes():
    boxes = np.array([[0, 0, 10, 10]], dtype=np.float32)
    assert np.isclose(ious(boxes, boxes)[0, 0], 1.0)


def test_iou_disjoint_boxes():
    a = np.array([[0, 0, 10, 10]], dtype=np.float32)
    b = np.array([[100, 100, 110, 110]], dtype=np.float32)
    assert ious(a, b)[0, 0] == 0.0


def test_iou_half_overlap():
    a = np.array([[0, 0, 10, 10]], dtype=np.float32)
    b = np.array([[5, 0, 15, 10]], dtype=np.float32)
    # intersection 50, union 150 -> 1/3
    assert np.isclose(ious(a, b)[0, 0], 1.0 / 3.0, atol=1e-5)


def test_iou_empty_inputs():
    empty = np.empty((0, 4), dtype=np.float32)
    boxes = np.array([[0, 0, 10, 10]], dtype=np.float32)
    assert ious(empty, boxes).shape == (0, 1)
    assert ious(boxes, empty).shape == (1, 0)


def test_linear_assignment_perfect_match():
    # Two tracks, two detections, cost is low on the diagonal.
    cost = np.array([[0.0, 0.9], [0.9, 0.0]], dtype=np.float32)
    matches, ua, ub = linear_assignment(cost, thresh=0.5)
    assert sorted(matches.tolist()) == [[0, 0], [1, 1]]
    assert ua == () and ub == ()


def test_linear_assignment_threshold_rejects():
    cost = np.array([[0.9]], dtype=np.float32)
    matches, ua, ub = linear_assignment(cost, thresh=0.5)
    assert matches.shape == (0, 2)
    assert ua == (0,) and ub == (0,)


def test_linear_assignment_empty():
    cost = np.empty((0, 3), dtype=np.float32)
    matches, ua, ub = linear_assignment(cost, thresh=0.5)
    assert matches.shape == (0, 2)
    assert ua == () and ub == (0, 1, 2)


def test_iou_distance_from_arrays():
    a = [np.array([0, 0, 10, 10], dtype=np.float32)]
    b = [np.array([0, 0, 10, 10], dtype=np.float32)]
    d = iou_distance(a, b)
    assert np.isclose(d[0, 0], 0.0)


def test_fuse_score_scales_by_confidence():
    class _Det:
        def __init__(self, score):
            self.score = score

    cost = np.array([[0.0]], dtype=np.float32)  # perfect IoU
    fused = fuse_score(cost, [_Det(0.5)])
    # 1 - (iou_sim * score) = 1 - (1 * 0.5) = 0.5
    assert np.isclose(fused[0, 0], 0.5)
