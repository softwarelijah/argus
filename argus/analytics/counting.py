"""Line-crossing and zone-occupancy counting over tracks."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from .trajectory import track_centroid


def _side(a: np.ndarray, b: np.ndarray, p: np.ndarray) -> float:
    """Signed side of point ``p`` relative to the directed line ``a -> b``.

    Positive on the left, negative on the right (image coordinates).
    """
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


class LineCounter:
    """Count tracks crossing a directed line segment, by direction.

    A crossing is registered when a track's centroid changes side of the line
    between consecutive observations. Direction is reported as ``up`` (positive
    to negative side) or ``down`` (negative to positive).
    """

    def __init__(self, point_a, point_b) -> None:
        self.a = np.asarray(point_a, dtype=np.float32)
        self.b = np.asarray(point_b, dtype=np.float32)
        self.up = 0
        self.down = 0
        self._last_side: dict[int, float] = {}
        self._counted: set[int] = set()

    def update(self, tracks) -> None:
        for t in tracks:
            p = np.asarray(track_centroid(t), dtype=np.float32)
            side = _side(self.a, self.b, p)
            prev = self._last_side.get(t.track_id)
            self._last_side[t.track_id] = side
            if prev is None or side == 0 or prev == 0:
                continue
            # Sign change means the centroid crossed the line.
            if (prev > 0) != (side > 0):
                if prev > 0 and side < 0:
                    self.up += 1
                else:
                    self.down += 1
                self._counted.add(t.track_id)

    @property
    def total(self) -> int:
        return self.up + self.down

    def counts(self) -> dict[str, int]:
        return {"up": self.up, "down": self.down, "total": self.total}


def point_in_polygon(point, polygon) -> bool:
    """Ray-casting point-in-polygon test. ``polygon`` is a list of (x, y)."""
    x, y = point
    poly = np.asarray(polygon, dtype=np.float64)
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        intersect = ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi
        )
        if intersect:
            inside = not inside
        j = i
    return inside


class ZoneCounter:
    """Track occupancy and unique entries for a polygonal zone."""

    def __init__(self, polygon, name: str = "zone") -> None:
        self.polygon = [tuple(p) for p in polygon]
        self.name = name
        self.unique_entries = 0
        self.current_ids: set[int] = set()
        self._inside: set[int] = set()
        self._dwell: dict[int, int] = defaultdict(int)

    def update(self, tracks) -> None:
        present = set()
        for t in tracks:
            if point_in_polygon(track_centroid(t), self.polygon):
                present.add(t.track_id)
                self._dwell[t.track_id] += 1
                if t.track_id not in self._inside:
                    self.unique_entries += 1
        self._inside = present
        self.current_ids = present

    @property
    def occupancy(self) -> int:
        return len(self.current_ids)

    def dwell_frames(self, track_id: int) -> int:
        return self._dwell.get(track_id, 0)

    def counts(self) -> dict[str, int]:
        return {"occupancy": self.occupancy, "unique_entries": self.unique_entries}
