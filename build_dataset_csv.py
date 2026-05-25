"""Scan ./data for interior-segmentation datasets and build a pair CSV.

Looks for directory names containing 'interior_segmentation' under --data-root,
expects each to contain:
  <dataset>/{raw,raw_images}/<name>.{jpg,png,...}
  <dataset>/masks/<name>.{png,jpg,...}

For every image with a matching mask, writes a row to the CSV with columns:
  image_path, mask_path, depth_path, dataset

Paths in the CSV are relative to --data-root (so `kw2552/raw/foo.jpg`, not
`data/kw2552/raw/foo.jpg`); the dataloader prepends --data_path at load time.
Pass --abs to write absolute paths instead.

`depth_path` is left empty when no sibling depth directory is found. Once
pseudo-depth maps have been generated (DAM_V2/Depth-prepare.ipynb) into
`<dataset>/depth/`, re-run this script to fill the column in.

Run:
    python build_dataset_csv.py
    python build_dataset_csv.py --data-root ./data/interior_segmentation \
        --out ./data/interior_segmentation/index.csv --include interior_segmentation
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff")
IMAGE_DIR_NAMES = ("raw_images", "raw", "images")
MASK_DIR_NAMES = ("masks",)
DEPTH_DIR_NAMES = ("depth", "depth_large", "depth_base", "depth_small")


def find_image_dir(dataset_dir: Path) -> Path | None:
    for name in IMAGE_DIR_NAMES:
        cand = dataset_dir / name
        if cand.is_dir():
            return cand
    return None


def find_match(stem: str, search_dir: Path) -> Path | None:
    for ext in IMAGE_EXTS:
        cand = search_dir / f"{stem}{ext}"
        if cand.exists():
            return cand
    return None


def find_depth(stem: str, dataset_dir: Path) -> Path | None:
    for d in DEPTH_DIR_NAMES:
        depth_dir = dataset_dir / d
        if depth_dir.is_dir():
            m = find_match(stem, depth_dir)
            if m is not None:
                return m
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default="data/interior_segmentation", type=Path)
    ap.add_argument("--out", default="data/interior_segmentation/index.csv", type=Path)
    ap.add_argument("--include", default="interior_segmentation",
                    help="substring that must appear in the dataset folder name "
                         "(default: 'interior_segmentation')")
    ap.add_argument("--abs", action="store_true",
                    help="write absolute paths (default: relative to --data-root)")
    args = ap.parse_args()

    data_root = args.data_root.resolve()
    if not data_root.is_dir():
        raise SystemExit(f"data root not found: {data_root}")

    datasets = sorted(
        d for d in data_root.iterdir()
        if d.is_dir() and args.include in d.name
    )
    if not datasets:
        raise SystemExit(
            f"No datasets matching '{args.include}' under {data_root}."
        )

    rows = []
    summary = []
    missing_masks_total = 0
    for ds in datasets:
        img_dir = find_image_dir(ds)
        mask_dir = ds / "masks"
        if img_dir is None or not mask_dir.is_dir():
            print(f"[skip] {ds.name}: image_dir={img_dir} mask_dir_exists={mask_dir.is_dir()}")
            continue

        n_img = n_pair = n_missing = 0
        for p in sorted(img_dir.iterdir()):
            if p.suffix.lower() not in IMAGE_EXTS:
                continue
            n_img += 1
            m = find_match(p.stem, mask_dir)
            if m is None:
                n_missing += 1
                continue
            d = find_depth(p.stem, ds)
            ipath = p if args.abs else p.relative_to(data_root)
            mpath = m if args.abs else m.relative_to(data_root)
            dpath = ""
            if d is not None:
                dpath = str(d if args.abs else d.relative_to(data_root))
            rows.append({
                "image_path": str(ipath),
                "mask_path": str(mpath),
                "depth_path": dpath,
                "dataset": ds.name,
            })
            n_pair += 1
        missing_masks_total += n_missing
        summary.append((ds.name, n_img, n_pair, n_missing))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image_path", "mask_path", "depth_path", "dataset"])
        w.writeheader()
        w.writerows(rows)

    print()
    print(f"{'dataset':<46}  {'imgs':>5}  {'paired':>6}  {'no_mask':>7}")
    print("-" * 72)
    for name, ni, np_, nm in summary:
        print(f"{name:<46}  {ni:>5}  {np_:>6}  {nm:>7}")
    print("-" * 72)
    print(f"{'TOTAL':<46}  {sum(s[1] for s in summary):>5}  "
          f"{len(rows):>6}  {missing_masks_total:>7}")
    n_with_depth = sum(1 for r in rows if r["depth_path"])
    print(f"\nrows with depth_path populated: {n_with_depth} / {len(rows)}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
