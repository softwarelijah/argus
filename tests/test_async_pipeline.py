import numpy as np

from argus.detection.detector import Detections
from argus.pipeline import AsyncVideoPipeline, frame_source
from argus.tracking import TrackerConfig


class _CenterDetector:
    names = {0: "obj"}

    def __init__(self):
        self.t = 0

    def detect(self, frame):
        self.t += 1
        cx = 100 + 3 * self.t
        box = np.array([[cx - 15, 200, cx + 15, 230]], dtype=np.float32)
        return Detections(box, np.array([0.9], np.float32), np.array([0], np.float32), self.names)


def _frames(n):
    return [np.zeros((240, 320, 3), dtype=np.uint8) for _ in range(n)]


def test_frame_source_from_iterable():
    frames = _frames(5)
    out = list(frame_source(frames))
    assert len(out) == 5


def test_async_pipeline_processes_all_frames():
    pipe = AsyncVideoPipeline(_CenterDetector(), TrackerConfig(frame_rate=30), draw=False)
    results = list(pipe.run(_frames(12)))
    assert len(results) == 12
    assert results[-1].frame_id == 12


def test_async_pipeline_keeps_stable_id():
    pipe = AsyncVideoPipeline(_CenterDetector(), TrackerConfig(frame_rate=30), draw=False)
    ids = []
    for result in pipe.run(_frames(15)):
        ids.extend(t.track_id for t in result.tracks)
    assert len(set(ids)) == 1


def test_async_pipeline_max_frames():
    pipe = AsyncVideoPipeline(_CenterDetector(), TrackerConfig(frame_rate=30), draw=False)
    results = list(pipe.run(_frames(50), max_frames=10))
    assert len(results) == 10


def test_async_pipeline_surfaces_reader_errors():
    def _bad_source():
        yield np.zeros((240, 320, 3), dtype=np.uint8)
        raise ValueError("decode failed")

    pipe = AsyncVideoPipeline(_CenterDetector(), TrackerConfig(frame_rate=30), draw=False)
    try:
        list(pipe.run(_bad_source()))
    except ValueError as exc:
        assert "decode failed" in str(exc)
        return
    raise AssertionError("expected the reader error to propagate")
