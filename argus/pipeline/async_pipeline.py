"""Threaded pipeline that overlaps video decode with detection and tracking.

Reading and decoding frames (especially from RTSP / H.264) is a serial cost
that otherwise stalls the GPU. A dedicated reader thread fills a bounded queue
while the main thread runs detect + track, so decode and inference overlap and
the effective throughput approaches the slower of the two stages rather than
their sum.

For live sources, ``realtime=True`` drops stale frames when the consumer falls
behind, trading completeness for low latency (you always process the newest
frame, never a growing backlog).
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Iterable, Iterator
from typing import Callable

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore[assignment]

from ..tracking import TrackerConfig
from .video_pipeline import PipelineResult, VideoPipeline

_SENTINEL = object()


def frame_source(source) -> Iterator[np.ndarray]:
    """Yield BGR frames from a path/index/URL (via OpenCV) or an iterable."""
    if isinstance(source, Iterable) and not isinstance(source, (str, bytes)):
        yield from source
        return
    if cv2 is None:  # pragma: no cover
        raise ImportError("opencv-python is required to read video sources")
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"could not open video source: {source!r}")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
    finally:
        cap.release()


class _ReaderThread(threading.Thread):
    """Background thread that decodes frames into a bounded queue."""

    def __init__(self, source, maxsize: int, realtime: bool) -> None:
        super().__init__(daemon=True)
        self.source = source
        self.realtime = realtime
        self.queue: queue.Queue = queue.Queue(maxsize=maxsize)
        self._stop = threading.Event()
        self.error: Exception | None = None

    def run(self) -> None:
        try:
            for frame in frame_source(self.source):
                if self._stop.is_set():
                    break
                if self.realtime and self.queue.full():
                    try:
                        self.queue.get_nowait()  # drop the oldest frame
                    except queue.Empty:
                        pass
                self.queue.put(frame)
        except Exception as exc:  # surface reader errors to the consumer
            self.error = exc
        finally:
            self.queue.put(_SENTINEL)

    def stop(self) -> None:
        self._stop.set()


class AsyncVideoPipeline:
    """Drive a :class:`VideoPipeline` with a background decode thread."""

    def __init__(
        self,
        detector,
        tracker_config: TrackerConfig | None = None,
        names: dict[int, str] | None = None,
        draw: bool = True,
        reid_extractor=None,
        queue_size: int = 8,
        realtime: bool = False,
    ) -> None:
        self.pipeline = VideoPipeline(
            detector, tracker_config, names=names, draw=draw, reid_extractor=reid_extractor
        )
        self.queue_size = queue_size
        self.realtime = realtime

    @property
    def meter(self):
        return self.pipeline.meter

    def run(
        self,
        source,
        on_frame: Callable[[PipelineResult], None] | None = None,
        max_frames: int | None = None,
    ) -> Iterator[PipelineResult]:
        reader = _ReaderThread(source, self.queue_size, self.realtime)
        reader.start()

        frame_id = 0
        try:
            while True:
                item = reader.queue.get()
                if item is _SENTINEL:
                    break
                frame_id += 1
                result = self.pipeline.process_frame(item, frame_id)
                if on_frame is not None:
                    on_frame(result)
                yield result
                if max_frames is not None and frame_id >= max_frames:
                    break
        finally:
            reader.stop()
        if reader.error is not None:
            raise reader.error
