"""Download and unpack the VisDrone-DET dataset.

Usage:
    python scripts/download_visdrone.py --root datasets/VisDrone

VisDrone is distributed as Google Drive zips. The official ids are filled in
below; if Google Drive rate-limits, download the zips manually from
https://github.com/VisDrone/VisDrone-Dataset and pass --skip-download to just
unpack files already present under --root.
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

# Official VisDrone-DET2019 release archives.
ARCHIVES = {
    "VisDrone2019-DET-train.zip": "https://github.com/VisDrone/VisDrone-Dataset",
    "VisDrone2019-DET-val.zip": "https://github.com/VisDrone/VisDrone-Dataset",
    "VisDrone2019-DET-test-dev.zip": "https://github.com/VisDrone/VisDrone-Dataset",
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
