"""Offline linear interpolation to fill short gaps in tracking results.

When a track is briefly lost (occlusion) the tracker emits no box for those
frames even though the identity is later recovered. Linearly interpolating the
box across the gap recovers those frames offline, which lowers false negatives
and raises MOTA. Only gaps up to ``max_gap`` frames are filled, so genuinely
absent objects are not invented.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np


def interpolate_tracks(
    pred: dict[int, np.ndarray], max_gap: int = 20
) -> dict[int, np.ndarray]:
    """Fill short gaps in a ``{frame: (N,5)}`` result of ``[id,x1,y1,x2,y2]``.

    Returns a new result dict with interpolated boxes added. Inputs are not
    mutated.
    """
    # Collect each track id's boxes keyed by frame.
    by_id: dict[int, dict[int, np.ndarray]] = defaultdict(dict)
    for frame, rows in pred.items():
        for row in np.asarray(rows, dtype=np.float32).reshape(-1, 5):
            by_id[int(row[0])][frame] = row[1:]

    out: dict[int, list[list[float]]] = defaultdict(list)
    for frame, rows in pred.items():
        for row in np.asarray(rows, dtype=np.float32).reshape(-1, 5):
            out[frame].append(row.tolist())

    for tid, frames_boxes in by_id.items():
        frames = sorted(frames_boxes)
        for f0, f1 in zip(frames, frames[1:]):
            gap = f1 - f0
            if gap <= 1 or gap - 1 > max_gap:
                continue
            box0, box1 = frames_boxes[f0], frames_boxes[f1]
            for k in range(1, gap):
                alpha = k / gap
                interp = (1 - alpha) * box0 + alpha * box1
                out[f0 + k].append([float(tid), *interp.tolist()])

    return {f: np.asarray(rows, dtype=np.float32).reshape(-1, 5) for f, rows in out.items()}
