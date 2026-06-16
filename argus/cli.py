"""Argus command line interface.

Subcommands:
  track     run detection + tracking over a video source
  export    export YOLOv8 weights to ONNX and build a TensorRT engine
  prepare   convert a VisDrone download into YOLO training format
  benchmark measure end-to-end FPS and per-stage latency
"""

from __future__ import annotations

import argparse
import sys


def _build_detector(args):
    """Pick the detector backend from CLI args, optionally wrapped in SAHI."""
    if args.engine:
        from .data.visdrone import names_dict
        from .inference.trt_detector import TRTDetector

        detector = TRTDetector(
            args.engine, imgsz=args.imgsz, conf=args.conf, names=names_dict()
        )
    else:
        from .detection.detector import YOLODetector

        detector = YOLODetector(
            args.weights, conf=args.conf, imgsz=args.imgsz, device=args.device
        )

    if getattr(args, "sahi", False):
        from .detection.sahi import SlicedDetector

        detector = SlicedDetector(
            detector, slice_h=args.slice, slice_w=args.slice, overlap=args.slice_overlap
        )
    return detector


def cmd_track(args) -> int:
    import cv2

    from .pipeline import VideoPipeline
    from .tracking import TrackerConfig

    detector = _build_detector(args)

    reid_extractor = None
    if args.reid:
        from .inference.reid import ReIDExtractor

        reid_extractor = ReIDExtractor(device=args.device)

    config = TrackerConfig(
        track_thresh=args.track_thresh,
        frame_rate=args.fps,
        gmc_method=args.gmc,
        with_reid=args.reid,
    )
    pipeline = VideoPipeline(
        detector, config, draw=not args.no_draw, reid_extractor=reid_extractor
    )

    writer = None
    for result in pipeline.run(args.source, max_frames=args.max_frames):
        if args.output:
            if writer is None:
                h, w = result.frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(args.output, fourcc, args.fps, (w, h))
            writer.write(result.frame)
        if args.show:
            cv2.imshow("argus", result.frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    return 0


def cmd_export(args) -> int:
    from .inference.export import build_engine, export_onnx

    onnx_path = export_onnx(args.weights, args.onnx, imgsz=args.imgsz)
    print(f"exported ONNX: {onnx_path}")

    if args.engine:
        calib = None
        if args.precision == "int8":
            from .data.calibration import build_calibration_set

            calib = build_calibration_set(args.calib_images, imgsz=args.imgsz)
            print(f"built calibration set: {calib.shape}")
        engine_path = build_engine(
            onnx_path,
            args.engine,
            precision=args.precision,
            imgsz=args.imgsz,
            calib_frames=calib,
        )
        print(f"built TensorRT engine ({args.precision}): {engine_path}")
    return 0


def cmd_prepare(args) -> int:
    from .data.visdrone import convert_split, write_data_yaml

    total = 0
    for split in ("VisDrone2019-DET-train", "VisDrone2019-DET-val", "VisDrone2019-DET-test-dev"):
        split_dir = f"{args.root}/{split}"
        try:
            n = convert_split(args.root, split_dir)
        except FileNotFoundError:
            print(f"skipping missing split: {split}")
            continue
        print(f"converted {n} images in {split}")
        total += n
    yaml_path = write_data_yaml(args.root, args.out_yaml)
    print(f"wrote dataset yaml: {yaml_path} ({total} images total)")
    return 0


def cmd_benchmark(args) -> int:
    from .pipeline import VideoPipeline

    detector = _build_detector(args)
    pipeline = VideoPipeline(detector, draw=False)
    for _ in pipeline.run(args.source, max_frames=args.max_frames):
        pass
    summary = pipeline.meter.summary()
    print("benchmark results:")
    for key, value in summary.items():
        print(f"  {key:>12}: {value:8.2f}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="argus", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p):
        p.add_argument("--weights", default="yolov8s.pt", help="YOLOv8 .pt weights")
        p.add_argument("--engine", default=None, help="TensorRT .engine (overrides weights)")
        p.add_argument("--imgsz", type=int, default=1280)
        p.add_argument("--conf", type=float, default=0.25)
        p.add_argument("--device", default=None, help="cuda device, e.g. '0' or 'cpu'")
        p.add_argument("--sahi", action="store_true", help="enable SAHI sliced inference")
        p.add_argument("--slice", type=int, default=640, help="SAHI tile size")
        p.add_argument("--slice-overlap", type=float, default=0.2, help="SAHI tile overlap")

    p_track = sub.add_parser("track", help="run detection + tracking")
    add_common(p_track)
    p_track.add_argument("source", help="video file, webcam index or stream URL")
    p_track.add_argument("--output", default=None, help="write annotated mp4 here")
    p_track.add_argument("--show", action="store_true", help="display a live window")
    p_track.add_argument("--no-draw", action="store_true", help="skip annotation")
    p_track.add_argument("--track-thresh", type=float, default=0.5)
    p_track.add_argument("--fps", type=int, default=30)
    p_track.add_argument("--max-frames", type=int, default=None)
    p_track.add_argument(
        "--gmc", choices=["none", "orb", "ecc"], default="none",
        help="global motion compensation for moving cameras",
    )
    p_track.add_argument("--reid", action="store_true", help="enable appearance Re-ID")
    p_track.set_defaults(func=cmd_track)

    p_export = sub.add_parser("export", help="export to ONNX / TensorRT")
    p_export.add_argument("--weights", required=True)
    p_export.add_argument("--onnx", default=None)
    p_export.add_argument("--engine", default=None)
    p_export.add_argument("--precision", choices=["fp32", "fp16", "int8"], default="int8")
    p_export.add_argument("--imgsz", type=int, default=1280)
    p_export.add_argument("--calib-images", default=None, help="dir of calibration frames")
    p_export.set_defaults(func=cmd_export)

    p_prep = sub.add_parser("prepare", help="convert VisDrone to YOLO format")
    p_prep.add_argument("--root", required=True, help="VisDrone dataset root")
    p_prep.add_argument("--out-yaml", default="configs/visdrone.yaml")
    p_prep.set_defaults(func=cmd_prepare)

    p_bench = sub.add_parser("benchmark", help="measure FPS and latency")
    add_common(p_bench)
    p_bench.add_argument("source", help="video file or stream")
    p_bench.add_argument("--max-frames", type=int, default=300)
    p_bench.set_defaults(func=cmd_benchmark)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
