"""Generate a hard synthetic MOT sequence for evaluating the tracker.

Produces ground-truth trajectories plus a noisy detection stream with the
failure modes a real detector exhibits: missed detections (occlusion / blur),
localisation jitter, and clutter (false positives). Running the tracker on the
detections and scoring against the ground truth yields real MOTA / IDF1 numbers
on a non-trivial sequence, entirely on CPU.
"""

from __future__ import annotations

import numpy as np

W, H = 1280, 720


def generate_mot_scene(
    num_targets: int = 30,
    num_frames: int = 200,
    miss_rate: float = 0.12,
    clutter_per_frame: float = 2.0,
    jitter_std: float = 2.0,
    seed: int = 0,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Return ``(ground_truth, detections)`` for one synthetic sequence.

    ``ground_truth`` maps ``frame -> (N, 5)`` as ``[id, x1, y1, x2, y2]``.
    ``detections`` maps ``frame -> (M, 6)`` as ``[x1, y1, x2, y2, score, cls]``.
    Targets appear and disappear at staggered times to exercise births/deaths.
    """
    rng = np.random.default_rng(seed)

    speeds = rng.uniform(3, 8, size=num_targets)
    angles = rng.uniform(0, 2 * np.pi, size=num_targets)
    start_x = rng.uniform(0, W, size=num_targets)
    start_y = rng.uniform(0, H, size=num_targets)
    births = rng.integers(0, max(1, num_frames // 3), size=num_targets)
    lifespans = rng.integers(num_frames // 2, num_frames, size=num_targets)
    size = 32

    gt: dict[int, list[list[float]]] = {f: [] for f in range(1, num_frames + 1)}
    det: dict[int, list[list[float]]] = {f: [] for f in range(1, num_frames + 1)}

    for i in range(num_targets):
        tid = i + 1
        vx, vy = speeds[i] * np.cos(angles[i]), speeds[i] * np.sin(angles[i])
        x, y = start_x[i], start_y[i]
        for frame in range(1, num_frames + 1):
            age = frame - births[i]
            if age < 0 or age > lifespans[i]:
                continue
            x += vx
            y += vy
            if x < 0 or x > W:
                vx = -vx
                x = min(max(x, 0), W)
            if y < 0 or y > H:
                vy = -vy
                y = min(max(y, 0), H)

            half = size / 2
            gt[frame].append([tid, x - half, y - half, x + half, y + half])

            if rng.random() < miss_rate:
                continue  # missed detection
            j = rng.normal(0, jitter_std, 4)
            det[frame].append(
                [x - half + j[0], y - half + j[1], x + half + j[2], y + half + j[3],
                 float(rng.uniform(0.4, 0.95)), 3.0]
            )

    # Clutter: random low-confidence false positives.
    for frame in range(1, num_frames + 1):
        for _ in range(rng.poisson(clutter_per_frame)):
            cx, cy = rng.uniform(0, W), rng.uniform(0, H)
            det[frame].append([cx - 8, cy - 8, cx + 8, cy + 8, float(rng.uniform(0.2, 0.45)), 3.0])

    gt_arr = {f: np.asarray(v, dtype=np.float32).reshape(-1, 5) for f, v in gt.items()}
    det_arr = {f: np.asarray(v, dtype=np.float32).reshape(-1, 6) for f, v in det.items()}
    return gt_arr, det_arr


def run_tracker(detections: dict[int, np.ndarray], tracker) -> dict[int, np.ndarray]:
    """Run a tracker over a detection stream into a ``{frame: (N,5)}`` result."""
    pred: dict[int, np.ndarray] = {}
    for frame in sorted(detections):
        tracks = tracker.update(detections[frame])
        pred[frame] = np.asarray(
            [[t.track_id, *t.tlbr.tolist()] for t in tracks], dtype=np.float32
        ).reshape(-1, 5)
    return pred
