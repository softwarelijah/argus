"""Evaluate a trained YOLOv8 model on the VisDrone validation split.

Usage:
    python scripts/evaluate.py --weights runs/visdrone/yolov8s-1280/weights/best.pt

Reports mAP@50 and mAP@50-95 over the 10 VisDrone classes.
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", default="configs/visdrone.yaml")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--split", default="val")
    args = parser.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.weights)
    metrics = model.val(data=args.data, imgsz=args.imgsz, split=args.split, verbose=True)
    print(f"mAP@50:    {metrics.box.map50:.4f}")
    print(f"mAP@50-95: {metrics.box.map:.4f}")
    for i, name in enumerate(model.names.values()):
        if i < len(metrics.box.maps):
            print(f"  {name:18s} mAP@50-95: {metrics.box.maps[i]:.4f}")


if __name__ == "__main__":
    main()
