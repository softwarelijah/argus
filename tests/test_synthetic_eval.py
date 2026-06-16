"""Regression test: the tracker scores sanely on a hard synthetic sequence."""

import numpy as np

from argus.eval import evaluate, generate_mot_scene, run_tracker
from argus.tracking import ByteTracker, TrackerConfig


def test_generate_scene_shapes():
    gt, det = generate_mot_scene(num_targets=10, num_frames=50, seed=0)
    assert len(gt) == 50
    assert len(det) == 50
    # ground truth rows are [id, x1, y1, x2, y2]
    some = next(v for v in gt.values() if len(v))
    assert some.shape[1] == 5
    # detections are [x1, y1, x2, y2, score, cls]
    some_d = next(v for v in det.values() if len(v))
    assert some_d.shape[1] == 6


def test_tracker_scores_above_floor():
    gt, det = generate_mot_scene(num_targets=30, num_frames=200, seed=0)
    tracker = ByteTracker(TrackerConfig(frame_rate=30))
    pred = run_tracker(det, tracker)
    result = evaluate(gt, pred, iou_thresh=0.5)
    # hard sequence (12% miss + clutter): expect solid but imperfect scores
    assert result.mota > 0.7
    assert result.idf1 > 0.6
    assert result.mostly_tracked >= 15
    assert result.recall > 0.7


def test_results_are_deterministic():
    gt1, det1 = generate_mot_scene(num_targets=12, num_frames=40, seed=7)
    gt2, det2 = generate_mot_scene(num_targets=12, num_frames=40, seed=7)
    for f in gt1:
        assert np.array_equal(gt1[f], gt2[f])
        assert np.array_equal(det1[f], det2[f])
