"""End-to-end real-time detection and tracking pipeline.

Wires a detector (YOLOv8 PyTorch or TensorRT) to the ByteTrack tracker over an
arbitrary video source and optionally renders / writes the annotated stream.
The detector is duck-typed: anything with a ``detect(frame) -> Detections``
method works, so the PyTorch and TensorRT backends are interchangeable.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Callable

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore[assignment]

from ..tracking import ByteTracker, TrackerConfig
from ..utils.timer import FPSMeter
from ..utils.visualization import draw_hud, draw_tracks


@dataclass
class PipelineResult:
    """Per-frame output of the pipeline."""

    frame_id: int
    frame: np.ndarray
    tracks: list
    fps: float
    latency_ms: float


class VideoPipeline:
    """Stream frames through detection and tracking."""

    def __init__(
        self,
        detector,
        tracker_config: TrackerConfig | None = None,
        names: dict[int, str] | None = None,
        draw: bool = True,
        reid_extractor=None,
    ) -> None:
        self.detector = detector
        self.config = tracker_config or TrackerConfig()
        self.tracker = ByteTracker(self.config)
        self.names = names or getattr(detector, "names", {}) or {}
        self.draw = draw
        self.reid_extractor = reid_extractor

        # GMC needs the frame; Re-ID needs both the frame and per-box crops.
        self._needs_frame = self.config.gmc_method != "none"
        self.meter = FPSMeter()

    def process_frame(self, frame: np.ndarray, frame_id: int) -> PipelineResult:
        with self.meter.stage("detect"):
            detections = self.detector.detect(frame)

        embeddings = None
        if self.config.with_reid and self.reid_extractor is not None and len(detections) > 0:
            with self.meter.stage("reid"):
                embeddings = self.reid_extractor.extract(frame, detections.boxes)

        with self.meter.stage("track"):
            tracks = self.tracker.update(
                detections.to_array(),
                frame=frame if self._needs_frame else None,
                embeddings=embeddings,
            )

        self.meter.tick()
        latency = self.meter.latency_ms("detect") + self.meter.latency_ms("track")

        if self.draw:
            draw_tracks(frame, tracks, self.names)
            draw_hud(frame, self.meter.fps, len(tracks), latency)

        return PipelineResult(frame_id, frame, tracks, self.meter.fps, latency)

    def run(
        self,
        source: str | int,
        on_frame: Callable[[PipelineResult], None] | None = None,
        max_frames: int | None = None,
    ) -> Iterator[PipelineResult]:
        """Iterate over a video source yielding per-frame results.

        ``source`` is anything OpenCV's ``VideoCapture`` accepts: a file path,
        a webcam index, or an RTSP/HTTP stream URL.
        """
        if cv2 is None:  # pragma: no cover
            raise ImportError("opencv-python is required to run the pipeline")

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"could not open video source: {source!r}")

        frame_id = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame_id += 1
                result = self.process_frame(frame, frame_id)
                if on_frame is not None:
                    on_frame(result)
                yield result
                if max_frames is not None and frame_id >= max_frames:
                    break
        finally:
            cap.release()
