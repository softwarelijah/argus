"""Convert a VisDrone-DET download into YOLO training format.

Usage:
    python scripts/prepare_visdrone.py --root datasets/VisDrone

Expects the standard VisDrone layout under --root:
    VisDrone2019-DET-train/{images,annotations}
    VisDrone2019-DET-val/{images,annotations}
    VisDrone2019-DET-test-dev/{images,annotations}

Writes a YOLO labels/ directory in each split and a dataset YAML.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--out-yaml", default="configs/visdrone.yaml")
    args = parser.parse_args()

    from argus.data.visdrone import convert_split, write_data_yaml

    splits = [
        "VisDrone2019-DET-train",
        "VisDrone2019-DET-val",
        "VisDrone2019-DET-test-dev",
    ]
    total = 0
    for split in splits:
        split_dir = Path(args.root) / split
        if not split_dir.exists():
            print(f"skip (missing): {split}")
            continue
        n = convert_split(args.root, split_dir)
        print(f"{split}: converted {n} images")
        total += n

    yaml_path = write_data_yaml(args.root, args.out_yaml)
    print(f"dataset yaml -> {yaml_path}")
    print(f"total images: {total}")


if __name__ == "__main__":
    main()
