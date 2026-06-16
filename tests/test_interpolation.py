import numpy as np

from argus.eval import evaluate, generate_mot_scene, interpolate_tracks, run_tracker
from argus.tracking import ByteTracker, TrackerConfig


def test_interpolation_fills_gap():
    # track 1 present at frame 1 and 4, missing 2 and 3
    pred = {
        1: np.array([[1, 0, 0, 10, 10]], dtype=np.float32),
        4: np.array([[1, 30, 0, 40, 10]], dtype=np.float32),
    }
    out = interpolate_tracks(pred, max_gap=20)
    assert 2 in out and 3 in out
    # frame 2 should be 1/3 of the way from x=0 to x=30 -> x1 ~ 10
    row2 = out[2][0]
    assert row2[0] == 1
    assert abs(row2[1] - 10.0) < 1e-3
    row3 = out[3][0]
    assert abs(row3[1] - 20.0) < 1e-3


def test_interpolation_respects_max_gap():
    pred = {
        1: np.array([[1, 0, 0, 10, 10]], dtype=np.float32),
        50: np.array([[1, 100, 0, 110, 10]], dtype=np.float32),
    }
    out = interpolate_tracks(pred, max_gap=10)
    # gap of 48 frames is too long; nothing filled
    assert 25 not in out


def test_interpolation_preserves_existing():
    pred = {
        1: np.array([[1, 0, 0, 10, 10]], dtype=np.float32),
        2: np.array([[1, 5, 0, 15, 10]], dtype=np.float32),
    }
    out = interpolate_tracks(pred)
    assert np.array_equal(out[1], pred[1])
    assert np.array_equal(out[2], pred[2])


def test_interpolation_improves_or_holds_recall():
    gt, det = generate_mot_scene(num_targets=20, num_frames=150, seed=3)
    tracker = ByteTracker(TrackerConfig(frame_rate=30))
    pred = run_tracker(det, tracker)
    base = evaluate(gt, pred, iou_thresh=0.5)
    interp = evaluate(gt, interpolate_tracks(pred, max_gap=20), iou_thresh=0.5)
    # interpolation should not reduce recall and typically increases it
    assert interp.recall >= base.recall - 1e-6
