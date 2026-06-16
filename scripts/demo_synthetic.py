"""Run the tracker on a synthetic aerial scene (no model or GPU required).

Simulates N targets moving across a frame, emits noisy detections with random
misses and clutter, then feeds them through ByteTracker. This exercises the
full association stack and reports identity-preservation metrics. If OpenCV is
available it also writes an annotated mp4.

Usage:
    python scripts/demo_synthetic.py --targets 60 --frames 300 --output demo.mp4
"""

from __future__ import annotations

import argparse
import math

import numpy as np

from argus.tracking import ByteTracker, TrackerConfig


class _Target:
    """A single ground-truth target on a linear-ish trajectory."""

    def __init__(self, rng: np.random.Generator, w: int, h: int) -> None:
        self.x = rng.uniform(0, w)
        self.y = rng.uniform(0, h)
        speed = rng.uniform(1.5, 5.0)
        angle = rng.uniform(0, 2 * math.pi)
        self.vx = speed * math.cos(angle)
        self.vy = speed * math.sin(angle)
        self.size = rng.uniform(18, 40)  # small objects, aerial scale
        self.w, self.h = w, h

    def step(self) -> None:
        self.x += self.vx
        self.y += self.vy
        # Bounce off the frame edges to keep targets in view.
        if self.x < 0 or self.x > self.w:
            self.vx *= -1
            self.x = min(max(self.x, 0), self.w)
        if self.y < 0 or self.y > self.h:
            self.vy *= -1
            self.y = min(max(self.y, 0), self.h)

    def box(self) -> list[float]:
        half = self.size / 2
        return [self.x - half, self.y - half, self.x + half, self.y + half]


def simulate(num_targets: int, num_frames: int, seed: int = 0):
    """Yield (frame_id, detections) where detections is (M, 6)."""
    rng = np.random.default_rng(seed)
    W, H = 1280, 720
    targets = [_Target(rng, W, H) for _ in range(num_targets)]

    for frame_id in range(num_frames):
        dets = []
        for t in targets:
            t.step()
            if rng.random() < 0.10:  # 10 percent miss rate (occlusion / blur)
                continue
            x1, y1, x2, y2 = t.box()
            jitter = rng.normal(0, 1.5, size=4)
            score = rng.uniform(0.35, 0.95)
            cls = 3  # 'car'
            box = [x1 + jitter[0], y1 + jitter[1], x2 + jitter[2], y2 + jitter[3]]
            dets.append([*box, score, cls])

        # Sprinkle in a few low-confidence false positives (clutter).
        for _ in range(rng.integers(0, 4)):
            cx, cy = rng.uniform(0, W), rng.uniform(0, H)
            dets.append([cx - 10, cy - 10, cx + 10, cy + 10, rng.uniform(0.15, 0.35), 3])

        yield frame_id, (W, H), np.asarray(dets, dtype=np.float32).reshape(-1, 6)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=int, default=60)
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default=None, help="optional annotated mp4")
    args = parser.parse_args()

    tracker = ByteTracker(TrackerConfig(track_thresh=0.5, frame_rate=30))

    writer = None
    if args.output:
        try:
            import cv2

            from argus.utils.visualization import draw_hud, draw_tracks
        except ImportError:
            print("opencv not available, skipping video output")
            args.output = None

    peak_tracks = 0
    total_active = 0
    seen_ids: set[int] = set()
    frames_done = 0

    for _frame_id, (W, H), dets in simulate(args.targets, args.frames, args.seed):
        tracks = tracker.update(dets)
        peak_tracks = max(peak_tracks, len(tracks))
        total_active += len(tracks)
        seen_ids.update(t.track_id for t in tracks)
        frames_done += 1

        if args.output:
            import cv2

            from argus.utils.visualization import draw_hud, draw_tracks

            frame = np.full((H, W, 3), 30, dtype=np.uint8)
            draw_tracks(frame, tracks, {3: "car"})
            draw_hud(frame, 30.0, len(tracks))
            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(args.output, fourcc, 30, (W, H))
            writer.write(frame)

    if writer is not None:
        writer.release()
        print(f"wrote {args.output}")

    avg_active = total_active / max(frames_done, 1)
    print(f"targets simulated:     {args.targets}")
    print(f"frames processed:      {frames_done}")
    print(f"peak active tracks:    {peak_tracks}")
    print(f"avg active tracks:     {avg_active:.1f}")
    print(f"unique track ids:      {len(seen_ids)}")
    print(f"id fragmentation:      {len(seen_ids) / args.targets:.2f}x ids per target")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
