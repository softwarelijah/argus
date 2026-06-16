"""Benchmark end-to-end FPS and per-stage latency.

Usage:
    # on a real video
    python scripts/benchmark.py video.mp4 --engine weights/yolov8s-int8.engine
    python scripts/benchmark.py video.mp4 --weights yolov8s.pt --device 0

    # no video needed: feed synthetic frames of the deployment resolution
    python scripts/benchmark.py --synthetic 300 --engine weights/yolov8s-int8.engine
    python scripts/benchmark.py --synthetic 300 --onnx weights/yolov8n.onnx

Reports detection latency, tracking latency and overall FPS averaged over the
processed frames. Use it to verify the sub-30 ms / 30+ FPS targets on the
deployment device. Synthetic mode measures pure compute (no decode), which is
the right number for the latency claim.
"""

from __future__ import annotations

import argparse
import time


def _build_detector(args):
    if args.engine:
        from argus.data.visdrone import names_dict
        from argus.inference.trt_detector import TRTDetector

        return TRTDetector(args.engine, imgsz=args.imgsz, conf=args.conf, names=names_dict()), \
            f"tensorrt:{args.engine}"
    if args.onnx:
        from argus.data.visdrone import names_dict
        from argus.inference.onnx_detector import ORTDetector

        return ORTDetector(args.onnx, imgsz=args.imgsz, conf=args.conf, names=names_dict()), \
            f"onnx:{args.onnx}"
    from argus.detection.detector import YOLODetector

    return YOLODetector(args.weights, conf=args.conf, imgsz=args.imgsz, device=args.device), \
        f"yolo:{args.weights}"


def _run_synthetic(pipeline, n, width, height, warmup):
    import numpy as np

    rng = np.random.default_rng(0)
    frames = 0
    for i in range(n + warmup):
        frame = rng.integers(0, 255, size=(height, width, 3), dtype=np.uint8)
        pipeline.process_frame(frame, i + 1)
        frames += 1
        if frames == warmup:
            pipeline.meter = type(pipeline.meter)()
    return frames


def _run_video(pipeline, source, max_frames, warmup):
    frames = 0
    for _ in pipeline.run(source, max_frames=max_frames):
        frames += 1
        if frames == warmup:
            pipeline.meter = type(pipeline.meter)()
    return frames


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", nargs="?", default=None, help="video file or stream")
    parser.add_argument("--synthetic", type=int, default=0, help="benchmark on N random frames")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--weights", default="yolov8s.pt")
    parser.add_argument("--engine", default=None)
    parser.add_argument("--onnx", default=None)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=20)
    args = parser.parse_args()

    if not args.synthetic and not args.source:
        parser.error("provide a video source or --synthetic N")

    detector, backend = _build_detector(args)

    from argus.pipeline import VideoPipeline

    pipeline = VideoPipeline(detector, draw=False)

    start = time.perf_counter()
    if args.synthetic:
        frames = _run_synthetic(pipeline, args.synthetic, args.width, args.height, args.warmup)
        mode = f"synthetic {args.width}x{args.height}"
    else:
        frames = _run_video(pipeline, args.source, args.max_frames, args.warmup)
        mode = f"video:{args.source}"
    wall = time.perf_counter() - start

    summary = pipeline.meter.summary()
    print(f"backend: {backend}")
    print(f"mode:    {mode}")
    print(f"frames:  {frames}  ({wall:.1f}s wall)")
    for key, value in summary.items():
        unit = "fps" if key == "fps" else "ms"
        print(f"  {key:>12}: {value:8.2f} {unit}")


if __name__ == "__main__":
    main()
