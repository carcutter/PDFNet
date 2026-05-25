"""Download the pretrained PDFNet checkpoint released by the authors.

Fetches the DIS-5K-TR checkpoint folder linked in README.md
(https://drive.google.com/drive/folders/1dqkFVR4TElSRFNHhu6er45OQkoHhJsZz)
into ./checkpoints/. The folder also contains the Swin-B backbone weights
required by SwinB(pretrained=True) on first model build.

Usage:
    pip install gdown
    python download_checkpoint.py
    python download_checkpoint.py --dest /custom/path
"""
import argparse
import os
import sys
from pathlib import Path

PDFNET_DRIVE_FOLDER_ID = "1dqkFVR4TElSRFNHhu6er45OQkoHhJsZz"
SWIN_B_URL = (
    "https://github.com/SwinTransformer/storage/releases/download/"
    "v1.0.0/swin_base_patch4_window12_384_22k.pth"
)


def download_pdfnet(dest: Path):
    try:
        import gdown
    except ImportError:
        sys.exit("gdown is required: pip install gdown")
    dest.mkdir(parents=True, exist_ok=True)
    url = f"https://drive.google.com/drive/folders/{PDFNET_DRIVE_FOLDER_ID}"
    print(f"Downloading PDFNet checkpoint folder into {dest} ...")
    gdown.download_folder(url, output=str(dest), quiet=False, use_cookies=False)


def download_swin(dest: Path):
    import urllib.request
    dest.mkdir(parents=True, exist_ok=True)
    out = dest / "swin_base_patch4_window12_384_22k.pth"
    if out.exists():
        print(f"Swin-B backbone already present: {out}")
        return
    print(f"Downloading Swin-B backbone -> {out} ...")
    urllib.request.urlretrieve(SWIN_B_URL, out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", default="checkpoints", type=Path,
                        help="directory to download into (default: ./checkpoints)")
    parser.add_argument("--skip-pdfnet", action="store_true",
                        help="skip the PDFNet weights download")
    parser.add_argument("--skip-swin", action="store_true",
                        help="skip the Swin-B backbone download")
    args = parser.parse_args()

    if not args.skip_swin:
        download_swin(args.dest)
    if not args.skip_pdfnet:
        download_pdfnet(args.dest)

    print("\nDone. Expected files now under", args.dest.resolve())
    for p in sorted(args.dest.rglob("*.pth")):
        print(" ", p.relative_to(args.dest))


if __name__ == "__main__":
    main()
