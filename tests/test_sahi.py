import numpy as np

from argus.detection.detector import Detections
from argus.detection.sahi import SlicedDetector, generate_slices


def test_generate_slices_covers_image():
    slices = generate_slices(1000, 1000, slice_h=400, slice_w=400, overlap=0.2)
    assert len(slices) > 1
    # union of tiles must reach the far corner
    assert max(s[2] for s in slices) == 1000
    assert max(s[3] for s in slices) == 1000
    # every tile is the requested size (edge tiles are clamped, not shrunk)
    for x1, y1, x2, y2 in slices:
        assert x2 - x1 == 400
        assert y2 - y1 == 400


def test_generate_slices_small_image_single_tile():
    slices = generate_slices(300, 300, slice_h=640, slice_w=640)
    assert len(slices) == 1


class _FakeDetector:
    """Returns one fixed detection at the centre of every tile it sees."""

    names = {0: "obj"}

    def detect(self, frame):
        h, w = frame.shape[:2]
        cx, cy = w / 2, h / 2
        box = np.array([[cx - 5, cy - 5, cx + 5, cy + 5]], dtype=np.float32)
        return Detections(box, np.array([0.9], np.float32), np.array([0], np.float32), self.names)


def test_sliced_detector_offsets_and_merges():
    frame = np.zeros((800, 800, 3), dtype=np.uint8)
    sahi = SlicedDetector(_FakeDetector(), slice_h=400, slice_w=400, include_full_frame=False)
    det = sahi.detect(frame)
    # one detection per tile, all distinct centres, so none are merged away
    slices = generate_slices(800, 800, 400, 400, 0.2)
    assert len(det) == len(slices)
    # boxes must be inside the full frame, not stuck at tile-local coordinates
    assert det.boxes[:, 2].max() <= 800
    assert det.boxes[:, 0].min() >= 0


def test_sliced_detector_empty_frame():
    class _Empty:
        names = {}

        def detect(self, frame):
            return Detections(
                np.empty((0, 4), np.float32),
                np.empty((0,), np.float32),
                np.empty((0,), np.float32),
                {},
            )

    sahi = SlicedDetector(_Empty(), slice_h=400, slice_w=400)
    det = sahi.detect(np.zeros((800, 800, 3), np.uint8))
    assert len(det) == 0
