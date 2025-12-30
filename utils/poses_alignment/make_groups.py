#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Group images from a folder into G groups with overlap o between adjacent groups.

Rules:
- Total image count N is determined by the number of images found in the input folder.
- Each group length is computed by L* = (N + (G-1)*o)/G, distributing the remainder
  to the first r groups so that lengths are integers and sum to N + (G-1)*o with overlaps.
- Start indices follow: start_{i+1} = start_i + length_i - o (1-based indexing for readability).

Output:
- Creates subfolders group1 ... groupG inside the output directory.
- By default creates symlinks to original images (fast and space-efficient).
  Use --copy to copy files instead.

Example:
    python group_images.py \
        --input ./images \
        --output ./experiments/grouped_alignment/group4 \
        --groups 4 \
        --overlap 10 \
        --copy
"""

import argparse
import os
import re
import shutil
from pathlib import Path
from typing import List, Tuple

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".gif", ".webp"}

def natural_key(s: str):
    """Return a key for natural sorting (splits numbers so 'img2' < 'img10')."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def list_images(input_dir: Path, exts: List[str]) -> List[Path]:
    """List images in input_dir filtered by extensions, sorted naturally."""
    exts = {e.lower() if e.startswith('.') else f".{e.lower()}" for e in exts} if exts else IMAGE_EXTS
    files = [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    files.sort(key=lambda p: natural_key(p.name))
    return files

def compute_group_lengths(N: int, G: int, o: int) -> List[int]:
    """
    Compute integer group lengths given total N, groups G, and overlap o.
    L* = (N + (G-1)*o)/G
    Let L_base = floor(L*), remainder r = (N + (G-1)*o) % G.
    First r groups get (L_base + 1), others get L_base.
    """
    if G <= 0:
        raise ValueError("groups (G) must be > 0")
    if N <= 0:
        raise ValueError("No images found to group (N=0)")
    if o < 0:
        raise ValueError("overlap (o) must be >= 0")
    # Check feasibility: the effective coverage must be at least N
    effective = G * 1 - (G - 1) * (o / max(1, N))  # not used, just sanity
    L_star = (N + (G - 1) * o) / G
    L_base = int(L_star)  # floor
    r = (N + (G - 1) * o) % G
    lengths = [(L_base + 1 if i < r else L_base) for i in range(G)]
    # Additional sanity: sum(lengths) - (G-1)*o should equal N
    covered = sum(lengths) - (G - 1) * o
    if covered != N:
        # Adjust if rounding issues occur (should not for integers)
        delta = N - covered
        i = 0
        while delta != 0 and 0 <= i < G:
            adj = 1 if delta > 0 else -1
            # Ensure each length remains >= 1 and overlap logic still holds in indexing stage
            if lengths[i] + adj >= 1:
                lengths[i] += adj
                delta -= adj
            i = (i + 1) % G
        # Final assert
        assert sum(lengths) - (G - 1) * o == N, "Failed to adjust group lengths to cover N"
    return lengths

def compute_ranges(lengths: List[int], overlap: int) -> List[Tuple[int, int]]:
    """
    Compute 1-based index ranges [(start, end), ...] for each group, given group lengths
    and overlap between adjacent groups.
    """
    ranges = []
    start = 1
    for i, L in enumerate(lengths):
        end = start + L - 1
        ranges.append((start, end))
        if i < len(lengths) - 1:
            start = start + L - overlap
    return ranges

def materialize_groups(files: List[Path], ranges: List[Tuple[int, int]], out_dir: Path, do_copy: bool, dry_run: bool):
    """
    Create group subfolders and populate them with symlinks or copies of files
    according to the 1-based ranges given.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for gi, (s, e) in enumerate(ranges, start=1):
        gdir = out_dir / f"group{gi}/images"
        gdir.mkdir(parents=True, exist_ok=True)
        # Clamp indices within [1, len(files)]
        s_clamped = max(1, min(s, len(files)))
        e_clamped = max(1, min(e, len(files)))
        if e_clamped < s_clamped:
            continue
        for idx in range(s_clamped, e_clamped + 1):
            src = files[idx - 1]
            dst = gdir / src.name
            if dry_run:
                action = "COPY" if do_copy else "SYMLINK"
                print(f"[DRY-RUN] {action}: {src} -> {dst}")
            else:
                if dst.exists():
                    # Avoid overwriting existing files; remove then recreate
                    dst.unlink()
                if do_copy:
                    shutil.copy2(src, dst)
                else:
                    # Create relative symlink when possible (nicer to move the folder)
                    try:
                        rel = os.path.relpath(src, gdir)
                        os.symlink(rel, dst)
                    except OSError:
                        # Fallback to absolute symlink if relative fails (e.g., on Windows without dev mode)
                        os.symlink(src, dst)

def main():
    parser = argparse.ArgumentParser(description="Group images from a folder into G groups with overlap.")
    parser.add_argument("--input", "-i", type=Path, required=True, help="Input folder containing images.")
    parser.add_argument("--output", "-o", type=Path, required=True, help="Output folder to create group subfolders.")
    parser.add_argument("--groups", "-g", type=int, default=4, help="Number of groups (G). Default: 4")
    parser.add_argument("--overlap", "-p", type=int, default=10, help="Overlap between adjacent groups (o). Default: 10")
    parser.add_argument("--extensions", "-e", nargs="*", default=None,
                        help="Image extensions to include (e.g., jpg png). Default: common image types.")
    parser.add_argument("--copy", action="store_true",
                        help="Copy files instead of creating symlinks (default is symlink).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print actions without creating any files.")
    args = parser.parse_args()

    if not args.input.exists() or not args.input.is_dir():
        raise FileNotFoundError(f"Input directory not found: {args.input}")

    files = list_images(args.input, args.extensions or [])
    N = len(files)
    if N == 0:
        raise RuntimeError(f"No images found in: {args.input}")

    # Compute group lengths and ranges (1-based indices)
    lengths = compute_group_lengths(N=N, G=args.groups, o=args.overlap)
    ranges = compute_ranges(lengths, args.overlap)

    # Summary
    print("=== Grouping Summary ===")
    print(f"Input dir : {args.input}")
    print(f"Output dir: {args.output}")
    print(f"Images    : {N}")
    print(f"Groups    : {args.groups}")
    print(f"Overlap   : {args.overlap}")
    print(f"Lengths   : {lengths}")
    print(f"Ranges(1-based): {ranges}")
    print(f"Mode      : {'COPY' if args.copy else 'SYMLINK'}")
    print(f"Dry-run   : {args.dry_run}")
    print("========================")

    # Materialize
    materialize_groups(files, ranges, args.output, do_copy=args.copy, dry_run=args.dry_run)

if __name__ == "__main__":
    main()
