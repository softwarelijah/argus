"""Demonstrate the analytics layer on a synthetic scene (no model or GPU).

Targets move across the frame through a counting line and a zone. The tracker
feeds a LineCounter, ZoneCounter and SpeedEstimator, and (if OpenCV is present)
an annotated video with trails, the line and the zone is written out.

Usage:
    python scripts/demo_analytics.py --targets 20 --frames 200 --output analytics.mp4
"""

from __future__ import annotations

import argparse

import numpy as np

from argus.analytics import LineCounter, SpeedEstimator, TrajectoryStore, ZoneCounter
from argus.tracking import ByteTracker, TrackerConfig

W, H = 1280, 720


def simulate(num_targets: int, num_frames: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    # Each target travels left to right at its own height and speed.
    ys = rng.uniform(100, H - 100, size=num_targets)
    speeds = rng.uniform(4, 9, size=num_targets)
    starts = rng.uniform(-300, 0, size=num_targets)
    size = 34

    for frame_id in range(1, num_frames + 1):
        dets = []
        for i in range(num_targets):
            x = starts[i] + speeds[i] * frame_id
            if x < 0 or x > W:
                continue
            y = ys[i]
            half = size / 2
            jitter = rng.normal(0, 1.0, 2)
            dets.append(
                [x - half + jitter[0], y - half + jitter[1], x + half, y + half, 0.9, 3]
            )
        yield frame_id, np.asarray(dets, dtype=np.float32).reshape(-1, 6)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=int, default=20)
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    tracker = ByteTracker(TrackerConfig(track_thresh=0.5, frame_rate=args.fps))
    trajectories = TrajectoryStore(max_len=45)
    line = LineCounter((W // 2, 0), (W // 2, H))  # vertical line mid-frame
    zone = ZoneCounter([(800, 150), (1100, 150), (1100, 550), (800, 550)], name="AOI")
    speedometer = SpeedEstimator(fps=args.fps)

    writer = None
    if args.output:
        try:
            import cv2

            from argus.utils.visualization import (
                draw_hud,
                draw_line,
                draw_tracks,
                draw_trails,
                draw_zone,
            )

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(args.output, fourcc, args.fps, (W, H))
        except ImportError:
            print("opencv not available, skipping video output")
            args.output = None

    max_speed = 0.0
    for frame_id, dets in simulate(args.targets, args.frames, args.seed):
        tracks = tracker.update(dets)
        trajectories.update(tracks, frame_id)
        line.update(tracks)
        zone.update(tracks)
        for t in tracks:
            max_speed = max(max_speed, speedometer.speed(trajectories, t.track_id))

        if args.output:
            import cv2

            from argus.utils.visualization import (
                draw_hud,
                draw_line,
                draw_tracks,
                draw_trails,
                draw_zone,
            )

            frame = np.full((H, W, 3), 30, dtype=np.uint8)
            draw_zone(frame, zone.polygon, label=f"{zone.name}: {zone.occupancy}")
            draw_line(frame, line.a, line.b, label=f"crossed: {line.total}")
            draw_trails(frame, trajectories, [t.track_id for t in tracks])
            draw_tracks(frame, tracks, {3: "car"})
            draw_hud(frame, float(args.fps), len(tracks))
            writer.write(frame)

    if writer is not None:
        writer.release()
        print(f"wrote {args.output}")

    print(f"line crossings:   {line.counts()}")
    print(f"zone entries:     {zone.unique_entries} unique, occupancy {zone.occupancy}")
    print(f"peak speed (px/s): {max_speed:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
