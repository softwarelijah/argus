"""Download and unpack the VisDrone-DET dataset.

Usage:
    python scripts/download_visdrone.py --root datasets/VisDrone

Downloads the VisDrone-DET2019 splits from the stable Ultralytics GitHub
release mirror (direct zip downloads, no Google Drive rate limits). If a
download fails, fetch the zips manually from
https://github.com/VisDrone/VisDrone-Dataset into --root and re-run with
--skip-download to just unpack them.
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

# VisDrone-DET2019 archives mirrored as GitHub release assets (direct download).
_BASE = "https://github.com/ultralytics/yolov5/releases/download/v1.0"
ARCHIVES = {
    "VisDrone2019-DET-train.zip": f"{_BASE}/VisDrone2019-DET-train.zip",
    "VisDrone2019-DET-val.zip": f"{_BASE}/VisDrone2019-DET-val.zip",
    "VisDrone2019-DET-test-dev.zip": f"{_BASE}/VisDrone2019-DET-test-dev.zip",
}


def _unpack(zip_path: Path, root: Path) -> None:
    print(f"unpacking {zip_path.name}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(root)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="datasets/VisDrone")
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)

    for name, url in ARCHIVES.items():
        zip_path = root / name
        if not args.skip_download and not zip_path.exists():
            print(f"downloading {name} ...")
            try:
                urlretrieve(url, zip_path)
            except Exception as exc:  # noqa: BLE001
                print(f"  could not auto-download ({exc}).")
                print(f"  download {name} manually into {root} and re-run with --skip-download.")
                continue
        if zip_path.exists():
            _unpack(zip_path, root)

    print(f"done. next: python scripts/prepare_visdrone.py --root {root}")


if __name__ == "__main__":
    main()
