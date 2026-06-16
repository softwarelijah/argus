"""Export trained YOLOv8 weights to an INT8 TensorRT engine.

Usage:
    python scripts/export_tensorrt.py \
        --weights runs/visdrone/yolov8s-1280/weights/best.pt \
        --calib-images datasets/VisDrone/VisDrone2019-DET-val/images \
        --precision int8

This is a thin wrapper around argus.inference.export so the full pipeline can
be reproduced from the command line. Run it on the GPU / Jetson host that has
the NVIDIA TensorRT stack installed.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--onnx", default=None)
    parser.add_argument("--engine", default=None)
    parser.add_argument("--precision", choices=["fp32", "fp16", "int8"], default="int8")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--calib-images", default=None)
    parser.add_argument("--num-calib", type=int, default=256)
    args = parser.parse_args()

    from argus.inference.export import build_engine, export_onnx

    stem = Path(args.weights).stem
    onnx_path = args.onnx or f"weights/{stem}.onnx"
    engine_path = args.engine or f"weights/{stem}-{args.precision}.engine"
    Path("weights").mkdir(exist_ok=True)

    onnx_path = export_onnx(args.weights, onnx_path, imgsz=args.imgsz)
    print(f"ONNX: {onnx_path}")

    calib = None
    if args.precision == "int8":
        if not args.calib_images:
            raise SystemExit("--calib-images is required for INT8 export")
        from argus.data.calibration import build_calibration_set

        calib = build_calibration_set(
            args.calib_images, num_samples=args.num_calib, imgsz=args.imgsz
        )
        print(f"calibration set: {calib.shape}")

    engine_path = build_engine(
        onnx_path, engine_path, precision=args.precision, imgsz=args.imgsz, calib_frames=calib
    )
    print(f"engine: {engine_path}")


if __name__ == "__main__":
    main()
