"""Per-track trajectory history, used for trails, speed and counting."""

from __future__ import annotations

from collections import defaultdict, deque

import numpy as np


def track_centroid(track) -> tuple[float, float]:
    """Return the (x, y) centroid of a track's current box."""
    x1, y1, x2, y2 = track.tlbr
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


class TrajectoryStore:
    """Rolling history of centroids per track id."""

    def __init__(self, max_len: int = 60) -> None:
        self.max_len = max_len
        self._history: dict[int, deque] = defaultdict(lambda: deque(maxlen=max_len))
        self._last_seen: dict[int, int] = {}

    def update(self, tracks, frame_id: int) -> None:
        for t in tracks:
            cx, cy = track_centroid(t)
            self._history[t.track_id].append((frame_id, cx, cy))
            self._last_seen[t.track_id] = frame_id

    def trail(self, track_id: int) -> list[tuple[float, float]]:
        """Return the (x, y) points for a track, oldest first."""
        return [(x, y) for _f, x, y in self._history.get(track_id, ())]

    def points(self, track_id: int) -> list[tuple[int, float, float]]:
        return list(self._history.get(track_id, ()))

    def prune(self, frame_id: int, max_age: int = 120) -> None:
        """Drop trajectories not seen for ``max_age`` frames."""
        stale = [
            tid for tid, last in self._last_seen.items() if frame_id - last > max_age
        ]
        for tid in stale:
            self._history.pop(tid, None)
            self._last_seen.pop(tid, None)

    def __contains__(self, track_id: int) -> bool:
        return track_id in self._history


def displacement(points: list[tuple[int, float, float]], window: int = 5) -> np.ndarray:
    """Pixel displacement vector over the last ``window`` samples."""
    if len(points) < 2:
        return np.zeros(2, dtype=np.float32)
    recent = points[-window:]
    (_, x0, y0), (_, x1, y1) = recent[0], recent[-1]
    return np.array([x1 - x0, y1 - y0], dtype=np.float32)
