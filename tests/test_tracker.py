import numpy as np

from argus.tracking import ByteTracker, TrackerConfig
from argus.tracking.basetrack import TrackState


def _box(cx, cy, size=30, score=0.9, cls=0):
    half = size / 2
    return [cx - half, cy - half, cx + half, cy + half, score, cls]


def test_single_target_keeps_stable_id():
    tracker = ByteTracker(TrackerConfig(frame_rate=30))
    ids = []
    for i in range(20):
        dets = np.array([_box(100 + 4 * i, 200)], dtype=np.float32)
        tracks = tracker.update(dets)
        if tracks:
            ids.append(tracks[0].track_id)
    # one continuous trajectory should hold a single id
    assert len(set(ids)) == 1
    assert len(ids) >= 18


def test_two_crossing_targets_get_distinct_ids():
    tracker = ByteTracker(TrackerConfig(frame_rate=30))
    final_ids = set()
    for i in range(25):
        a = _box(50 + 6 * i, 100)
        b = _box(400 - 6 * i, 100)
        tracks = tracker.update(np.array([a, b], dtype=np.float32))
        final_ids = {t.track_id for t in tracks}
    assert len(final_ids) >= 2


def test_low_score_detection_recovers_track():
    # ByteTrack's second stage should keep a track alive on a low-score box.
    cfg = TrackerConfig(track_thresh=0.5, new_track_thresh=0.6, frame_rate=30)
    tracker = ByteTracker(cfg)

    for i in range(5):
        tracker.update(np.array([_box(100 + 4 * i, 200, score=0.9)], dtype=np.float32))
    # now feed only a low-confidence detection
    tracks = tracker.update(np.array([_box(120, 200, score=0.3)], dtype=np.float32))
    assert len(tracks) == 1


def test_lost_track_is_reaped_after_buffer():
    cfg = TrackerConfig(track_buffer=5, frame_rate=30)
    tracker = ByteTracker(cfg)
    for i in range(5):
        tracker.update(np.array([_box(100 + 4 * i, 200)], dtype=np.float32))
    # feed empty frames longer than the buffer
    for _ in range(10):
        tracker.update(np.empty((0, 6), dtype=np.float32))
    active = [t for t in tracker.tracked_tracks if t.state == TrackState.Tracked]
    assert active == []


def test_many_targets_tracked_simultaneously():
    tracker = ByteTracker(TrackerConfig(frame_rate=30))
    rng = np.random.default_rng(0)
    centers = rng.uniform(50, 1200, size=(55, 2))
    vels = rng.uniform(-3, 3, size=(55, 2))

    peak = 0
    for _ in range(15):
        centers += vels
        dets = np.array([_box(cx, cy) for cx, cy in centers], dtype=np.float32)
        tracks = tracker.update(dets)
        peak = max(peak, len(tracks))
    # should comfortably track 50+ simultaneous targets
    assert peak >= 50


def test_empty_input_returns_no_tracks():
    tracker = ByteTracker()
    tracks = tracker.update(np.empty((0, 6), dtype=np.float32))
    assert tracks == []


def test_reset_clears_state_and_ids():
    tracker = ByteTracker(TrackerConfig(frame_rate=30))
    tracker.update(np.array([_box(100, 200)], dtype=np.float32))
    tracker.reset()
    assert tracker.frame_id == 0
    assert tracker.tracked_tracks == []
    tracks = tracker.update(np.array([_box(100, 200)], dtype=np.float32))
    # ids restart from 1 after a reset
    assert all(t.track_id == 1 for t in tracks) or tracks == []
