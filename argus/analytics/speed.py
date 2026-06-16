"""Per-track speed estimation in pixels/s or, with a homography, m/s."""

from __future__ import annotations

import numpy as np

from .trajectory import TrajectoryStore, displacement


class SpeedEstimator:
    """Estimate track speed from trajectory displacement.

    Without a homography, speed is reported in pixels per second. Supply a 3x3
    image-to-ground homography (``H``) and a metres-per-unit scale to convert
    image motion into ground-plane metres per second, the usual setup for
    nadir / oblique drone footage with a calibrated ground plane.
    """

    def __init__(
        self,
        fps: float = 30.0,
        homography: np.ndarray | None = None,
        meters_per_unit: float = 1.0,
        window: int = 5,
    ) -> None:
        self.fps = fps
        self.window = window
        self.meters_per_unit = meters_per_unit
        self.H = np.asarray(homography, dtype=np.float64) if homography is not None else None

    def _to_ground(self, pt: np.ndarray) -> np.ndarray:
        if self.H is None:
            return pt
        v = self.H @ np.array([pt[0], pt[1], 1.0])
        return np.array([v[0] / v[2], v[1] / v[2]])

    def speed(self, store: TrajectoryStore, track_id: int) -> float:
        """Return the current speed estimate for a track.

        Units are m/s when a homography is set, otherwise pixels/s.
        """
        points = store.points(track_id)
        if len(points) < 2:
            return 0.0
        recent = points[-self.window :]
        (f0, x0, y0), (f1, x1, y1) = recent[0], recent[-1]
        dframes = max(1, f1 - f0)

        if self.H is None:
            dist = float(np.linalg.norm(displacement(points, self.window)))
            per_frame = dist / dframes
        else:
            p0 = self._to_ground(np.array([x0, y0]))
            p1 = self._to_ground(np.array([x1, y1]))
            dist = float(np.linalg.norm(p1 - p0)) * self.meters_per_unit
            per_frame = dist / dframes

        return per_frame * self.fps
