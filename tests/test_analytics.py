import numpy as np

from argus.analytics import (
    LineCounter,
    SpeedEstimator,
    TrajectoryStore,
    ZoneCounter,
    point_in_polygon,
)


class _Track:
    """Minimal track stub exposing track_id and a tlbr box."""

    def __init__(self, tid, cx, cy, size=10):
        self.track_id = tid
        half = size / 2
        self._box = np.array([cx - half, cy - half, cx + half, cy + half], dtype=np.float32)

    @property
    def tlbr(self):
        return self._box


def test_trajectory_store_accumulates_points():
    store = TrajectoryStore(max_len=10)
    for f in range(1, 6):
        store.update([_Track(1, 10 * f, 20)], f)
    assert len(store.trail(1)) == 5
    assert store.trail(1)[0] == (10, 20)


def test_trajectory_prune_removes_stale():
    store = TrajectoryStore()
    store.update([_Track(1, 0, 0)], 1)
    store.prune(frame_id=200, max_age=120)
    assert 1 not in store


def test_line_counter_counts_crossing_direction():
    # vertical line at x=100 from (100,0) to (100,200)
    counter = LineCounter((100, 0), (100, 200))
    # track moves left -> right across the line
    for x in (80, 95, 105, 120):
        counter.update([_Track(1, x, 100)])
    assert counter.total == 1
    # one direction got the count, the other did not
    assert (counter.up == 1) ^ (counter.down == 1)


def test_line_counter_no_crossing_when_parallel():
    counter = LineCounter((100, 0), (100, 200))
    for y in (10, 50, 90):  # moves along the same side
        counter.update([_Track(1, 50, y)])
    assert counter.total == 0


def test_point_in_polygon():
    square = [(0, 0), (10, 0), (10, 10), (0, 10)]
    assert point_in_polygon((5, 5), square)
    assert not point_in_polygon((15, 5), square)


def test_zone_counter_occupancy_and_entries():
    zone = ZoneCounter([(0, 0), (100, 0), (100, 100), (0, 100)])
    zone.update([_Track(1, 50, 50), _Track(2, 200, 200)])  # only track 1 inside
    assert zone.occupancy == 1
    assert zone.unique_entries == 1
    zone.update([_Track(1, 50, 60)])  # still inside, no new entry
    assert zone.unique_entries == 1
    zone.update([_Track(1, 300, 300)])  # leaves
    assert zone.occupancy == 0
    assert zone.unique_entries == 1


def test_speed_estimator_pixels_per_second():
    store = TrajectoryStore()
    # move 10 px/frame along x; at 30 fps that is 300 px/s
    for f in range(1, 7):
        store.update([_Track(1, 10 * f, 0)], f)
    est = SpeedEstimator(fps=30.0, window=5)
    speed = est.speed(store, 1)
    assert abs(speed - 300.0) < 1.0


def test_speed_estimator_with_homography_scales():
    store = TrajectoryStore()
    for f in range(1, 7):
        store.update([_Track(1, 10 * f, 0)], f)
    # identity homography, 0.5 m per pixel-unit -> half the pixel speed
    est = SpeedEstimator(fps=30.0, window=5, homography=np.eye(3), meters_per_unit=0.5)
    speed = est.speed(store, 1)
    assert abs(speed - 150.0) < 1.0
