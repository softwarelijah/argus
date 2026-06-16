"""Benchmark end-to-end FPS and per-stage latency.

Usage:
    python scripts/benchmark.py --engine weights/yolov8s-int8.engine video.mp4
    python scripts/benchmark.py --weights yolov8s.pt --device 0 video.mp4

Reports detection latency, tracking latency and overall FPS averaged over the
processed frames. Use it to verify the sub-30 ms / 30+ FPS targets on the
deployment device.
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("--weights", default="yolov8s.pt")
    parser.add_argument("--engine", default=None)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=20)
    args = parser.parse_args()

    if args.engine:
        from argus.data.visdrone import names_dict
        from argus.inference.trt_detector import TRTDetector

        detector = TRTDetector(args.engine, imgsz=args.imgsz, conf=args.conf, names=names_dict())
        backend = f"tensorrt:{args.engine}"
    else:
        from argus.detection.detector import YOLODetector

        detector = YOLODetector(
            args.weights, conf=args.conf, imgsz=args.imgsz, device=args.device
        )
        backend = f"yolo:{args.weights}"

    from argus.pipeline import VideoPipeline

    pipeline = VideoPipeline(detector, draw=False)

    frames = 0
    for _ in pipeline.run(args.source, max_frames=args.max_frames):
        frames += 1
        if frames == args.warmup:
            # Reset meters after warmup so cold-start cost is excluded.
            pipeline.meter = type(pipeline.meter)()

    summary = pipeline.meter.summary()
    print(f"backend: {backend}")
    print(f"frames:  {frames}")
    for key, value in summary.items():
        unit = "fps" if key == "fps" else "ms"
        print(f"  {key:>12}: {value:8.2f} {unit}")


if __name__ == "__main__":
    main()
