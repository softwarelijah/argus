import numpy as np

from argus.detection.detector import Detections
from argus.pipeline import VideoPipeline
from argus.tracking import TrackerConfig


class _MovingDetector:
    """Emits one detection that drifts right each call."""

    names = {0: "obj"}

    def __init__(self):
        self.t = 0

    def detect(self, frame):
        self.t += 1
        cx = 100 + 4 * self.t
        box = np.array([[cx - 15, 200, cx + 15, 230]], dtype=np.float32)
        return Detections(box, np.array([0.9], np.float32), np.array([0], np.float32), self.names)


class _FakeReID:
    calls = 0

    def extract(self, frame, boxes):
        _FakeReID.calls += 1
        return np.ones((len(boxes), 8), dtype=np.float32)


def _frame():
    return np.zeros((480, 640, 3), dtype=np.uint8)


def test_pipeline_tracks_across_frames():
    pipe = VideoPipeline(_MovingDetector(), TrackerConfig(frame_rate=30), draw=False)
    ids = []
    for i in range(1, 16):
        result = pipe.process_frame(_frame(), i)
        ids.extend(t.track_id for t in result.tracks)
    assert len(set(ids)) == 1  # stable identity


def test_pipeline_reid_extractor_is_called():
    _FakeReID.calls = 0
    cfg = TrackerConfig(with_reid=True, frame_rate=30)
    pipe = VideoPipeline(_MovingDetector(), cfg, draw=False, reid_extractor=_FakeReID())
    for i in range(1, 5):
        pipe.process_frame(_frame(), i)
    assert _FakeReID.calls == 4


def test_pipeline_gmc_enabled_passes_frame():
    # With GMC on, process_frame must run without error on real frames.
    cfg = TrackerConfig(gmc_method="orb", frame_rate=30)
    pipe = VideoPipeline(_MovingDetector(), cfg, draw=False)
    rng = np.random.default_rng(0)
    for i in range(1, 6):
        frame = rng.integers(0, 255, size=(240, 320, 3), dtype=np.uint8)
        result = pipe.process_frame(frame, i)
    assert result.frame_id == 5


def test_pipeline_draw_overlays_without_error():
    pipe = VideoPipeline(_MovingDetector(), TrackerConfig(frame_rate=30), draw=True)
    result = pipe.process_frame(_frame(), 1)
    assert result.frame.shape == (480, 640, 3)
