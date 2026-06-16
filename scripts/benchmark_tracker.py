"""Benchmark the tracker's own throughput (detector excluded), on CPU.

Measures ByteTracker.update latency as a function of simultaneous targets, so
the "50+ targets at 30+ FPS" tracking claim can be substantiated independently
of the detector or any GPU.

Usage:
    python scripts/benchmark_tracker.py
    python scripts/benchmark_tracker.py --counts 10 25 50 100 --frames 300
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from argus.tracking import ByteTracker, TrackerConfig

W, H = 1280, 720


def _scene(num_targets, num_frames, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.uniform([40, 40], [W - 40, H - 40], size=(num_targets, 2))
    vels = rng.uniform(-4, 4, size=(num_targets, 2))
    for _ in range(num_frames):
        centers += vels
        centers = np.clip(centers, 0, [W, H])
        half = 16
        boxes = np.column_stack(
            [centers[:, 0] - half, centers[:, 1] - half,
             centers[:, 0] + half, centers[:, 1] + half]
        )
        scores = rng.uniform(0.5, 0.95, size=(num_targets, 1))
        cls = np.full((num_targets, 1), 3.0)
        yield np.concatenate([boxes, scores, cls], axis=1).astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counts", type=int, nargs="+", default=[10, 25, 50, 100, 200])
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=20)
    args = parser.parse_args()

    print(f"tracker throughput ({args.frames} frames per setting, CPU)")
    print(f"{'targets':>8} {'ms/frame':>10} {'FPS':>8}")
    for n in args.counts:
        tracker = ByteTracker(TrackerConfig(frame_rate=30))
        times = []
        for i, dets in enumerate(_scene(n, args.frames + args.warmup)):
            start = time.perf_counter()
            tracker.update(dets)
            elapsed = time.perf_counter() - start
            if i >= args.warmup:
                times.append(elapsed)
        ms = float(np.mean(times)) * 1000.0
        fps = 1000.0 / ms if ms > 0 else 0.0
        print(f"{n:>8} {ms:>10.2f} {fps:>8.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
