"""Read and write MOTChallenge-format tracking files.

MOTChallenge rows are:
    frame, id, bb_left, bb_top, bb_width, bb_height, conf, x, y, z

Boxes are stored top-left + width/height and converted here to the
``[track_id, x1, y1, x2, y2]`` per-frame layout the metrics expect.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_mot(path: str | Path, min_conf: float = 0.0) -> dict[int, np.ndarray]:
    """Load a MOTChallenge file into a ``{frame: (N, 5)}`` dict."""
    frames: dict[int, list[list[float]]] = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        frame = int(float(parts[0]))
        tid = int(float(parts[1]))
        x, y, w, h = (float(v) for v in parts[2:6])
        conf = float(parts[6]) if len(parts) > 6 and parts[6] != "" else 1.0
        if conf < min_conf:
            continue
        frames.setdefault(frame, []).append([tid, x, y, x + w, y + h])
    return {f: np.asarray(rows, dtype=np.float32) for f, rows in frames.items()}


def write_mot(path: str | Path, results: list[tuple[int, list]]) -> None:
    """Write tracker output to a MOTChallenge file.

    ``results`` is a list of ``(frame_id, tracks)`` where each track exposes
    ``track_id`` and a ``tlwh`` box.
    """
    lines = []
    for frame_id, tracks in results:
        for t in tracks:
            x, y, w, h = t.tlwh
            lines.append(
                f"{frame_id},{t.track_id},{x:.2f},{y:.2f},{w:.2f},{h:.2f},"
                f"{getattr(t, 'score', 1.0):.4f},-1,-1,-1"
            )
    Path(path).write_text("\n".join(lines))
