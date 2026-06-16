"""Fine-tune YOLOv8 on VisDrone.

Usage:
    python scripts/train.py --config configs/train.yaml

Reads hyperparameters from the YAML config and hands them to Ultralytics. The
defaults target small-object aerial detection: 1280 px input, late mosaic
closing and light copy-paste augmentation.
"""

from __future__ import annotations

import argparse

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    from ultralytics import YOLO

    model = YOLO(cfg.pop("model"))
    cfg["resume"] = args.resume
    results = model.train(**cfg)
    print("training complete")
    print(results)


if __name__ == "__main__":
    main()
