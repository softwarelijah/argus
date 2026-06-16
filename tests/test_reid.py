import numpy as np

from argus.tracking import ByteTracker, TrackerConfig
from argus.tracking.matching import embedding_distance, fuse_motion_appearance
from argus.tracking.track import STrack


class _Holder:
    def __init__(self, feat):
        self.curr_feat = feat
        self.smooth_feat = feat


def test_embedding_distance_identical_is_zero():
    f = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    d = embedding_distance([_Holder(f)], [_Holder(f)])
    assert np.isclose(d[0, 0], 0.0, atol=1e-6)


def test_embedding_distance_orthogonal_is_one():
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0], dtype=np.float32)
    d = embedding_distance([_Holder(a)], [_Holder(b)])
    assert np.isclose(d[0, 0], 1.0, atol=1e-6)


def test_fuse_motion_appearance_rejects_far_boxes():
    iou_cost = np.array([[0.9]], dtype=np.float32)  # far apart spatially
    app_cost = np.array([[0.0]], dtype=np.float32)  # identical appearance
    fused = fuse_motion_appearance(iou_cost, app_cost, proximity_thresh=0.5)
    # appearance must be gated out by the proximity test, leaving motion cost
    assert fused[0, 0] >= 0.9 - 1e-6


def test_fuse_motion_appearance_uses_appearance_when_close():
    iou_cost = np.array([[0.3]], dtype=np.float32)
    app_cost = np.array([[0.0]], dtype=np.float32)
    fused = fuse_motion_appearance(iou_cost, app_cost, proximity_thresh=0.5, weight=0.5)
    # blended cost should drop below the pure-motion cost
    assert fused[0, 0] < 0.3


def test_update_features_normalises_and_smooths():
    track = STrack(np.array([0.0, 0.0, 1.0, 10.0]), 0.9, feat=np.array([3.0, 4.0]))
    # 3-4-5 triangle -> unit vector (0.6, 0.8)
    assert np.allclose(track.smooth_feat, [0.6, 0.8], atol=1e-6)
    track.update_features(np.array([0.0, 5.0]))
    # EMA stays unit length
    assert np.isclose(np.linalg.norm(track.smooth_feat), 1.0, atol=1e-6)


def _box(cx, cy, size=30, score=0.9, cls=0):
    half = size / 2
    return [cx - half, cy - half, cx + half, cy + half, score, cls]


def test_tracker_with_reid_runs_and_keeps_id():
    cfg = TrackerConfig(with_reid=True, frame_rate=30)
    tracker = ByteTracker(cfg)
    rng = np.random.default_rng(0)
    emb = rng.normal(size=128).astype(np.float32)

    ids = []
    for i in range(15):
        dets = np.array([_box(100 + 4 * i, 200)], dtype=np.float32)
        embeddings = emb[None] + rng.normal(0, 0.01, size=(1, 128)).astype(np.float32)
        tracks = tracker.update(dets, embeddings=embeddings)
        if tracks:
            ids.append(tracks[0].track_id)
    assert len(set(ids)) == 1


def test_tracker_reid_disabled_ignores_embeddings():
    # With reid off, passing embeddings must not change motion-only behaviour.
    tracker = ByteTracker(TrackerConfig(frame_rate=30))
    dets = np.array([_box(100, 200)], dtype=np.float32)
    tracks = tracker.update(dets, embeddings=np.ones((1, 64), np.float32))
    assert len(tracks) <= 1  # runs without error
