"""Data association helpers (IoU cost and linear assignment)."""

from __future__ import annotations

import numpy as np

try:  # lap gives the fastest Jonker-Volgenant solver, but is optional.
    import lap

    _HAVE_LAP = True
except ImportError:  # pragma: no cover - exercised only without lap installed
    _HAVE_LAP = False
    from scipy.optimize import linear_sum_assignment


def linear_assignment(
    cost_matrix: np.ndarray, thresh: float
) -> tuple[np.ndarray, tuple, tuple]:
    """Solve the assignment problem.

    Returns ``(matches, unmatched_a, unmatched_b)`` where ``matches`` is an
    (M, 2) array of index pairs and the unmatched entries are index tuples.
    Pairs whose cost exceeds ``thresh`` are rejected.
    """
    if cost_matrix.size == 0:
        return (
            np.empty((0, 2), dtype=int),
            tuple(range(cost_matrix.shape[0])),
            tuple(range(cost_matrix.shape[1])),
        )

    matches, unmatched_a, unmatched_b = [], [], []
    if _HAVE_LAP:
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
        for ix, mx in enumerate(x):
            if mx >= 0:
                matches.append([ix, mx])
        unmatched_a = np.where(x < 0)[0]
        unmatched_b = np.where(y < 0)[0]
    else:
        rows, cols = linear_sum_assignment(cost_matrix)
        matched_a, matched_b = set(), set()
        for r, c in zip(rows, cols):
            if cost_matrix[r, c] <= thresh:
                matches.append([r, c])
                matched_a.add(r)
                matched_b.add(c)
        unmatched_a = [r for r in range(cost_matrix.shape[0]) if r not in matched_a]
        unmatched_b = [c for c in range(cost_matrix.shape[1]) if c not in matched_b]

    matches = np.asarray(matches, dtype=int).reshape(-1, 2)
    return matches, tuple(unmatched_a), tuple(unmatched_b)


def ious(atlbrs: np.ndarray, btlbrs: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two sets of (x1, y1, x2, y2) boxes."""
    atlbrs = np.asarray(atlbrs, dtype=np.float32).reshape(-1, 4)
    btlbrs = np.asarray(btlbrs, dtype=np.float32).reshape(-1, 4)
    if atlbrs.size == 0 or btlbrs.size == 0:
        return np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float32)

    area_a = (atlbrs[:, 2] - atlbrs[:, 0]) * (atlbrs[:, 3] - atlbrs[:, 1])
    area_b = (btlbrs[:, 2] - btlbrs[:, 0]) * (btlbrs[:, 3] - btlbrs[:, 1])

    lt = np.maximum(atlbrs[:, None, :2], btlbrs[None, :, :2])
    rb = np.minimum(atlbrs[:, None, 2:], btlbrs[None, :, 2:])
    wh = np.clip(rb - lt, a_min=0, a_max=None)
    inter = wh[..., 0] * wh[..., 1]

    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0).astype(np.float32)


def iou_distance(atracks: list, btracks: list) -> np.ndarray:
    """IoU based cost matrix (1 - IoU) between two lists of tracks/detections."""
    if atracks and isinstance(atracks[0], np.ndarray):
        atlbrs = atracks
    else:
        atlbrs = [t.tlbr for t in atracks]
    if btracks and isinstance(btracks[0], np.ndarray):
        btlbrs = btracks
    else:
        btlbrs = [t.tlbr for t in btracks]

    iou_matrix = ious(atlbrs, btlbrs)
    return 1.0 - iou_matrix


def fuse_score(cost_matrix: np.ndarray, detections: list) -> np.ndarray:
    """Fuse detection confidence into an IoU cost matrix.

    Weighting the IoU similarity by the detection score sharpens association
    when several tracks compete for overlapping boxes.
    """
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1.0 - cost_matrix
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    fused = iou_sim * det_scores
    return 1.0 - fused
