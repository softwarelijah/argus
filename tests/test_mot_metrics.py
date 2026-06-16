import numpy as np

from argus.eval.mot_metrics import evaluate


def _seq(tracks_per_frame):
    """Build a {frame: (N,5)} dict from a list of per-frame [(id,x1,y1,x2,y2)]."""
    return {
        f: np.asarray(rows, dtype=np.float32).reshape(-1, 5)
        for f, rows in enumerate(tracks_per_frame)
    }


def test_perfect_tracking_scores_one():
    frames = [[[1, 0, 0, 10, 10], [2, 50, 50, 60, 60]] for _ in range(10)]
    gt = _seq(frames)
    pred = _seq(frames)
    r = evaluate(gt, pred)
    assert np.isclose(r.mota, 1.0)
    assert np.isclose(r.idf1, 1.0)
    assert r.id_switches == 0
    assert r.fp == 0 and r.fn == 0
    assert r.mostly_tracked == 2


def test_all_false_negatives():
    gt = _seq([[[1, 0, 0, 10, 10]] for _ in range(5)])
    pred = {}
    r = evaluate(gt, pred)
    assert r.fn == 5
    assert r.tp == 0
    assert r.mota <= 0.0
    assert r.mostly_lost == 1


def test_false_positives_lower_mota():
    gt = _seq([[[1, 0, 0, 10, 10]] for _ in range(5)])
    # prediction adds a spurious extra box every frame
    pred = _seq([[[1, 0, 0, 10, 10], [9, 100, 100, 110, 110]] for _ in range(5)])
    r = evaluate(gt, pred)
    assert r.fp == 5
    assert r.fn == 0
    assert r.mota < 1.0


def test_id_switch_detected():
    # gt id 1 present all frames; prediction swaps id halfway through
    gt = _seq([[[1, 0, 0, 10, 10]] for _ in range(6)])
    pred_frames = [[[1, 0, 0, 10, 10]] for _ in range(3)] + [
        [[2, 0, 0, 10, 10]] for _ in range(3)
    ]
    pred = _seq(pred_frames)
    r = evaluate(gt, pred)
    assert r.id_switches == 1
    # IDF1 penalises the identity break even though boxes overlap perfectly
    assert r.idf1 < 1.0


def test_half_overlap_below_threshold_is_miss():
    gt = _seq([[[1, 0, 0, 10, 10]]])
    pred = _seq([[[1, 8, 0, 18, 10]]])  # small overlap, IoU < 0.5
    r = evaluate(gt, pred, iou_thresh=0.5)
    assert r.tp == 0
    assert r.fn == 1 and r.fp == 1
