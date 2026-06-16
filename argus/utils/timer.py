"""Lightweight timing utilities for measuring pipeline latency and FPS."""

from __future__ import annotations

import time
from collections import deque
from contextlib import contextmanager


class MovingAverage:
    """Fixed-window moving average over recent samples."""

    def __init__(self, window: int = 60) -> None:
        self._values: deque[float] = deque(maxlen=window)

    def add(self, value: float) -> None:
        self._values.append(value)

    @property
    def average(self) -> float:
        return sum(self._values) / len(self._values) if self._values else 0.0


class FPSMeter:
    """Tracks per-stage latency in milliseconds and overall FPS."""

    def __init__(self, window: int = 60) -> None:
        self._stages: dict[str, MovingAverage] = {}
        self._frame_ma = MovingAverage(window)
        self._window = window
        self._last_frame: float | None = None

    @contextmanager
    def stage(self, name: str):
        """Context manager timing a named stage, recording milliseconds."""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self._stages.setdefault(name, MovingAverage(self._window)).add(elapsed_ms)

    def tick(self) -> None:
        """Mark the end of a full frame for FPS calculation."""
        now = time.perf_counter()
        if self._last_frame is not None:
            self._frame_ma.add(now - self._last_frame)
        self._last_frame = now

    @property
    def fps(self) -> float:
        avg = self._frame_ma.average
        return 1.0 / avg if avg > 0 else 0.0

    def latency_ms(self, name: str) -> float:
        return self._stages[name].average if name in self._stages else 0.0

    def summary(self) -> dict[str, float]:
        out = {f"{name}_ms": ma.average for name, ma in self._stages.items()}
        out["fps"] = self.fps
        return out
