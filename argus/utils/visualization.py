"""Drawing helpers for detections and tracks."""

from __future__ import annotations

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - allows headless import
    cv2 = None


def _color_for_id(idx: int) -> tuple[int, int, int]:
    """Deterministic, well-separated BGR color for a track id."""
    idx = int(idx) * 3 + 1
    r = (37 * idx) % 255
    g = (17 * idx + 80) % 255
    b = (29 * idx + 160) % 255
    return int(b), int(g), int(r)


def draw_tracks(
    frame: np.ndarray,
    tracks,
    names: dict[int, str] | None = None,
    draw_score: bool = True,
) -> np.ndarray:
    """Draw track boxes, ids and class labels onto a BGR frame in place."""
    if cv2 is None:  # pragma: no cover
        raise ImportError("opencv-python is required for visualization")
    names = names or {}
    for track in tracks:
        x1, y1, x2, y2 = track.tlbr.astype(int)
        color = _color_for_id(track.track_id)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = names.get(track.cls, str(track.cls))
        text = f"#{track.track_id} {label}"
        if draw_score:
            text += f" {track.score:.2f}"

        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 2, y1), color, -1)
        cv2.putText(
            frame, text, (x1 + 1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
            cv2.LINE_AA,
        )
    return frame


def draw_trails(frame: np.ndarray, store, track_ids, thickness: int = 2) -> np.ndarray:
    """Draw fading trajectory trails for the given track ids."""
    if cv2 is None:  # pragma: no cover
        raise ImportError("opencv-python is required for visualization")
    for tid in track_ids:
        pts = store.trail(tid)
        if len(pts) < 2:
            continue
        color = _color_for_id(tid)
        for i in range(1, len(pts)):
            p0 = (int(pts[i - 1][0]), int(pts[i - 1][1]))
            p1 = (int(pts[i][0]), int(pts[i][1]))
            cv2.line(frame, p0, p1, color, thickness, cv2.LINE_AA)
    return frame


def draw_line(frame: np.ndarray, a, b, label: str = "", color=(0, 200, 255)) -> np.ndarray:
    """Draw a counting line with an optional label."""
    if cv2 is None:  # pragma: no cover
        raise ImportError("opencv-python is required for visualization")
    a = (int(a[0]), int(a[1]))
    b = (int(b[0]), int(b[1]))
    cv2.line(frame, a, b, color, 2, cv2.LINE_AA)
    if label:
        mid = ((a[0] + b[0]) // 2, (a[1] + b[1]) // 2)
        cv2.putText(frame, label, mid, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return frame


def draw_zone(frame: np.ndarray, polygon, label: str = "", color=(255, 120, 0)) -> np.ndarray:
    """Draw a polygonal zone outline with an optional label."""
    if cv2 is None:  # pragma: no cover
        raise ImportError("opencv-python is required for visualization")
    pts = np.asarray(polygon, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)
    if label:
        x, y = polygon[0]
        cv2.putText(
            frame, label, (int(x), int(y) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
            cv2.LINE_AA,
        )
    return frame


def draw_hud(frame: np.ndarray, fps: float, num_tracks: int, latency_ms: float = 0.0) -> np.ndarray:
    """Overlay a small heads-up display with FPS and active track count."""
    if cv2 is None:  # pragma: no cover
        raise ImportError("opencv-python is required for visualization")
    lines = [f"FPS: {fps:5.1f}", f"Tracks: {num_tracks}"]
    if latency_ms:
        lines.append(f"Latency: {latency_ms:4.1f} ms")
    y = 24
    for line in lines:
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(
            frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1, cv2.LINE_AA
        )
        y += 28
    return frame
