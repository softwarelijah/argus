"""Score the tracker on a hard synthetic MOT sequence (CPU, no GPU/dataset).

Generates a sequence with births/deaths, missed detections and clutter, runs
ByteTrack over the detections, and reports real CLEAR-MOT / IDF1 numbers from
the evaluation harness. Use it to sanity-check the tracker and the metrics, and
to show the eval stack producing sane numbers without downloading a dataset.

Usage:
    python scripts/eval_mot_demo.py --targets 30 --frames 200
    python scripts/eval_mot_demo.py --gmc orb   # compare with motion comp on
"""

from __future__ import annotations

import argparse


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=int, default=30)
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument("--miss-rate", type=float, default=0.12)
    parser.add_argument("--clutter", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gmc", choices=["none", "orb", "ecc"], default="none")
    args = parser.parse_args()

    from argus.eval import evaluate, generate_mot_scene, run_tracker
    from argus.tracking import ByteTracker, TrackerConfig

    gt, det = generate_mot_scene(
        num_targets=args.targets,
        num_frames=args.frames,
        miss_rate=args.miss_rate,
        clutter_per_frame=args.clutter,
        seed=args.seed,
    )
    tracker = ByteTracker(TrackerConfig(frame_rate=30, gmc_method=args.gmc))
    pred = run_tracker(det, tracker)
    result = evaluate(gt, pred, iou_thresh=0.5)

    print(f"synthetic MOT sequence: {args.targets} targets, {args.frames} frames")
    print(f"  detector miss rate: {args.miss_rate:.0%}, clutter ~{args.clutter}/frame")
    print("tracking metrics")
    print(f"  MOTA:        {result.mota * 100:6.2f}")
    print(f"  MOTP:        {result.motp * 100:6.2f}")
    print(f"  IDF1:        {result.idf1 * 100:6.2f}")
    print(f"  Recall:      {result.recall * 100:6.2f}")
    print(f"  Precision:   {result.precision * 100:6.2f}")
    print(f"  ID switches: {result.id_switches}")
    print(f"  MT / ML:     {result.mostly_tracked} / {result.mostly_lost} of {result.num_gt_ids}")
    print(f"  FP / FN:     {result.fp} / {result.fn}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
