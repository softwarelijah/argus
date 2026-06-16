"""Multi-object tracking metrics: CLEAR-MOT (MOTA/MOTP) and IDF1.

Implements the standard tracking metrics from scratch (numpy + scipy) so the
tracker can be evaluated without a heavyweight external dependency:

  - MOTA  = 1 - (FN + FP + IDSW) / num_gt        (accuracy)
  - MOTP  = mean IoU over true-positive matches  (localisation)
  - IDF1  = identity F1 over globally matched trajectories
  - plus FP, FN, ID switches, mostly-tracked / mostly-lost counts

Inputs are per-frame dicts mapping ``frame_id -> (N, 5)`` arrays laid out as
``[track_id, x1, y1, x2, y2]`` for both ground truth and predictions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class MOTResult:
    mota: float
    motp: float
    idf1: float
    idp: float
    idr: float
    precision: float
    recall: float
    num_gt: int
    num_pred: int
    tp: int
    fp: int
    fn: int
    id_switches: int
    mostly_tracked: int
    mostly_lost: int
    num_gt_ids: int

    def as_dict(self) -> dict[str, float]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def evaluate(
    gt: dict[int, np.ndarray],
    pred: dict[int, np.ndarray],
    iou_thresh: float = 0.5,
) -> MOTResult:
    """Compute CLEAR-MOT and IDF1 metrics for one sequence."""
    frames = sorted(set(gt) | set(pred))

    fp = fn = idsw = tp = 0
    matched_iou_sum = 0.0
    num_gt = 0

    # gt track id -> pred track id assigned in the previous matched frame
    last_match: dict[int, int] = {}

    # per-gt-id presence and hit counts for MT / ML and IDF1
    gt_presence: dict[int, int] = {}
    cooccur: dict[tuple[int, int], int] = {}
    gt_total: dict[int, int] = {}
    pred_total: dict[int, int] = {}

    for frame in frames:
        g = gt.get(frame, np.empty((0, 5), np.float32)).reshape(-1, 5)
        p = pred.get(frame, np.empty((0, 5), np.float32)).reshape(-1, 5)
        g_ids = g[:, 0].astype(int)
        p_ids = p[:, 0].astype(int)
        num_gt += len(g)

        for gid in g_ids:
            gt_presence[gid] = gt_presence.get(gid, 0) + 1
            gt_total[gid] = gt_total.get(gid, 0) + 1
        for pid in p_ids:
            pred_total[pid] = pred_total.get(pid, 0) + 1

        iou = _iou_matrix(g[:, 1:], p[:, 1:])

        matched_g: set[int] = set()
        matched_p: set[int] = set()

        # 1) Preserve existing identity assignments that are still valid.
        for gi, gid in enumerate(g_ids):
            pid = last_match.get(gid)
            if pid is None:
                continue
            pj = np.where(p_ids == pid)[0]
            if len(pj) == 1 and iou[gi, pj[0]] >= iou_thresh:
                pj = pj[0]
                matched_g.add(gi)
                matched_p.add(pj)
                tp += 1
                matched_iou_sum += iou[gi, pj]
                cooccur[(gid, pid)] = cooccur.get((gid, pid), 0) + 1

        # 2) Hungarian on the remaining detections.
        rem_g = [i for i in range(len(g_ids)) if i not in matched_g]
        rem_p = [j for j in range(len(p_ids)) if j not in matched_p]
        if rem_g and rem_p:
            sub = iou[np.ix_(rem_g, rem_p)]
            cost = 1.0 - sub
            rows, cols = linear_sum_assignment(cost)
            for r, c in zip(rows, cols):
                if sub[r, c] < iou_thresh:
                    continue
                gi, pj = rem_g[r], rem_p[c]
                gid, pid = int(g_ids[gi]), int(p_ids[pj])
                matched_g.add(gi)
                matched_p.add(pj)
                tp += 1
                matched_iou_sum += sub[r, c]
                cooccur[(gid, pid)] = cooccur.get((gid, pid), 0) + 1
                if gid in last_match and last_match[gid] != pid:
                    idsw += 1
                last_match[gid] = pid

        fn += len(g_ids) - len(matched_g)
        fp += len(p_ids) - len(matched_p)

    # Mostly tracked / mostly lost based on per-id hit ratio.
    hits: dict[int, int] = {}
    for (gid, _pid), c in cooccur.items():
        hits[gid] = hits.get(gid, 0) + c
    mt = ml = 0
    for gid, present in gt_presence.items():
        ratio = hits.get(gid, 0) / present if present else 0.0
        if ratio >= 0.8:
            mt += 1
        elif ratio <= 0.2:
            ml += 1

    mota = 1.0 - (fn + fp + idsw) / num_gt if num_gt else 0.0
    motp = matched_iou_sum / tp if tp else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    idf1, idp, idr = _compute_idf1(cooccur, gt_total, pred_total)

    return MOTResult(
        mota=mota,
        motp=motp,
        idf1=idf1,
        idp=idp,
        idr=idr,
        precision=precision,
        recall=recall,
        num_gt=num_gt,
        num_pred=sum(pred_total.values()),
        tp=tp,
        fp=fp,
        fn=fn,
        id_switches=idsw,
        mostly_tracked=mt,
        mostly_lost=ml,
        num_gt_ids=len(gt_presence),
    )


def _compute_idf1(
    cooccur: dict[tuple[int, int], int],
    gt_total: dict[int, int],
    pred_total: dict[int, int],
) -> tuple[float, float, float]:
    """Identity F1 via a global trajectory-to-trajectory matching.

    Maximises identity true positives (frames where a matched gt/pred pair
    co-occur above the IoU threshold) over a one-to-one assignment.
    """
    gt_ids = sorted(gt_total)
    pred_ids = sorted(pred_total)
    if not gt_ids or not pred_ids:
        return 0.0, 0.0, 0.0

    gi = {g: i for i, g in enumerate(gt_ids)}
    pj = {p: j for j, p in enumerate(pred_ids)}
    benefit = np.zeros((len(gt_ids), len(pred_ids)), dtype=np.float64)
    for (g, p), c in cooccur.items():
        benefit[gi[g], pj[p]] = c

    rows, cols = linear_sum_assignment(-benefit)
    idtp = int(benefit[rows, cols].sum())

    total_gt = sum(gt_total.values())
    total_pred = sum(pred_total.values())
    idfn = total_gt - idtp
    idfp = total_pred - idtp

    idp = idtp / (idtp + idfp) if (idtp + idfp) else 0.0
    idr = idtp / (idtp + idfn) if (idtp + idfn) else 0.0
    idf1 = 2 * idtp / (2 * idtp + idfp + idfn) if (2 * idtp + idfp + idfn) else 0.0
    return idf1, idp, idr
