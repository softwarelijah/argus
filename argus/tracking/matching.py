"""Data association helpers (IoU cost and linear assignment)."""

from __future__ import annotations

from collections.abc import Iterable

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

    matches: list[list[int]] = []
    unmatched_a: Iterable[int]
    unmatched_b: Iterable[int]
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

    matches_arr = np.asarray(matches, dtype=int).reshape(-1, 2)
    return matches_arr, tuple(unmatched_a), tuple(unmatched_b)


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

    iou_matrix = ious(np.asarray(atlbrs, dtype=np.float32), np.asarray(btlbrs, dtype=np.float32))
    return 1.0 - iou_matrix


def embedding_distance(tracks: list, detections: list, metric: str = "cosine") -> np.ndarray:
    """Appearance cost matrix between tracks and detections.

    Tracks use their smoothed (EMA) embedding; detections use their current
    embedding. Returns a (T, D) matrix of distances in [0, 2] for cosine.
    """
    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)
    if cost_matrix.size == 0:
        return cost_matrix

    det_feats = np.asarray([d.curr_feat for d in detections], dtype=np.float32)
    track_feats = np.asarray([t.smooth_feat for t in tracks], dtype=np.float32)

    if metric == "cosine":
        # Features are L2-normalised on update, so a dot product is the cosine
        # similarity; distance is 1 - similarity, clamped to be non-negative.
        sim = track_feats @ det_feats.T
        cost_matrix = np.maximum(0.0, 1.0 - sim)
    else:
        raise ValueError(f"unsupported embedding metric: {metric!r}")
    return cost_matrix.astype(np.float32)


def fuse_motion_appearance(
    iou_cost: np.ndarray,
    app_cost: np.ndarray,
    proximity_thresh: float = 0.5,
    appearance_thresh: float = 0.25,
    weight: float = 0.5,
) -> np.ndarray:
    """Fuse IoU (motion) and appearance costs into a single cost matrix.

    Appearance is only trusted when the boxes are also spatially plausible:
    pairs failing the IoU proximity gate or the appearance gate are set to a
    cost of 1 (rejected). Mirrors the BoT-SORT association rule.
    """
    if iou_cost.size == 0:
        return iou_cost
    app = app_cost.copy()
    app[iou_cost > proximity_thresh] = 1.0
    app[app_cost > appearance_thresh] = 1.0
    return np.minimum(iou_cost, weight * iou_cost + (1.0 - weight) * app)


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
