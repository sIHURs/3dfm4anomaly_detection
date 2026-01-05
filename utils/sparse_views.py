#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import random
from pathlib import Path
import sys


IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"
}


def is_image_file(p: Path) -> bool:
    return p.suffix.lower() in IMAGE_EXTENSIONS


def main():
    parser = argparse.ArgumentParser(
        description="Randomly keep a percentage of images in a folder and delete the rest."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to the folder containing images"
    )
    parser.add_argument(
        "--keep_percent",
        type=float,
        required=True,
        help="Percentage of images to keep (0 < percent <= 100)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility (default: 0)"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="If set, do not delete files, only print what would be deleted"
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    keep_percent = args.keep_percent

    if not input_dir.is_dir():
        print(f"[ERROR] {input_dir} is not a valid directory.")
        sys.exit(1)

    if not (0.0 < keep_percent <= 100.0):
        print("[ERROR] --keep_percent must be in the range (0, 100].")
        sys.exit(1)

    # Collect image files
    image_files = sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and is_image_file(p)
    )

    if len(image_files) == 0:
        print("[WARNING] No image files found.")
        sys.exit(0)

    # Number to keep
    num_keep = max(1, int(len(image_files) * keep_percent / 100.0))

    random.seed(args.seed)
    keep_files = set(random.sample(image_files, num_keep))
    remove_files = [p for p in image_files if p not in keep_files]

    print(f"[INFO] Total images: {len(image_files)}")
    print(f"[INFO] Keeping {num_keep} images ({keep_percent:.2f}%)")
    print(f"[INFO] Removing {len(remove_files)} images")

    # Delete files
    for p in remove_files:
        if args.dry_run:
            print(f"[DRY-RUN] Would remove: {p.name}")
        else:
            p.unlink()

    print("[DONE]")


if __name__ == "__main__":
    main()
