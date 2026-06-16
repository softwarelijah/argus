"""Evaluate tracking quality with CLEAR-MOT and IDF1 metrics.

Two modes:

  Compare an existing results file against ground truth:
    python scripts/eval_mot.py --gt gt.txt --pred results.txt

  Run the tracker on MOTChallenge public detections, then evaluate:
    python scripts/eval_mot.py --gt gt.txt --det det.txt --track-thresh 0.5

Both files are MOTChallenge format. Use --gt-min-conf 1 to keep only active
ground-truth boxes.
"""

from __future__ import annotations

import argparse

import numpy as np


def _run_tracker(det: dict, args) -> dict:
    from argus.tracking import ByteTracker, TrackerConfig

    tracker = ByteTracker(TrackerConfig(track_thresh=args.track_thresh, frame_rate=args.fps))
    results: dict[int, list[list[float]]] = {}
    for frame in sorted(det):
        rows = det[frame]  # (N, 5) [id, x1, y1, x2, y2]; public dets ignore id
        boxes = rows[:, 1:]
        scores = np.ones((len(rows), 1), dtype=np.float32)  # public dets are unscored
        cls = np.zeros((len(rows), 1), dtype=np.float32)
        dets = np.concatenate([boxes, scores, cls], axis=1)
        tracks = tracker.update(dets)
        results[frame] = [
            [t.track_id, *t.tlbr.tolist()] for t in tracks
        ]
    return {f: np.asarray(r, dtype=np.float32).reshape(-1, 5) for f, r in results.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", required=True)
    parser.add_argument("--pred", default=None, help="results file to evaluate")
    parser.add_argument("--det", default=None, help="public detections to track then evaluate")
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--gt-min-conf", type=float, default=0.0)
    parser.add_argument("--track-thresh", type=float, default=0.5)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    from argus.eval import evaluate, load_mot

    gt = load_mot(args.gt, min_conf=args.gt_min_conf)
    if args.pred:
        pred = load_mot(args.pred)
    elif args.det:
        pred = _run_tracker(load_mot(args.det), args)
    else:
        raise SystemExit("provide either --pred or --det")

    result = evaluate(gt, pred, iou_thresh=args.iou)

    print("tracking metrics")
    print(f"  MOTA:        {result.mota * 100:6.2f}")
    print(f"  MOTP:        {result.motp * 100:6.2f}")
    print(f"  IDF1:        {result.idf1 * 100:6.2f}")
    print(f"  IDP / IDR:   {result.idp * 100:6.2f} / {result.idr * 100:6.2f}")
    print(f"  Precision:   {result.precision * 100:6.2f}")
    print(f"  Recall:      {result.recall * 100:6.2f}")
    print(f"  ID switches: {result.id_switches}")
    print(f"  MT / ML:     {result.mostly_tracked} / {result.mostly_lost}")
    print(f"  FP / FN:     {result.fp} / {result.fn}")
    print(f"  GT / IDs:    {result.num_gt} dets, {result.num_gt_ids} ids")


if __name__ == "__main__":
    main()
