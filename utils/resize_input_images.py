#!/usr/bin/env python3
"""
Resize .jpg/.png images so that:
- If image is square: output is exactly 1024x1024
- If not square: keep aspect ratio, set the LONG side to 1024

Modes:
1) Output to new dir:  --out_dir /path/to/out
2) Overwrite in place: --inplace   (writes to temp then atomically replaces)
"""

import argparse
from pathlib import Path
from PIL import Image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}

def compute_new_size(w: int, h: int, long_side: int, square_size: int):
    if w == h:
        return square_size, square_size
    if w > h:
        new_w = long_side
        new_h = max(1, round(h * (long_side / w)))
    else:
        new_h = long_side
        new_w = max(1, round(w * (long_side / h)))
    return new_w, new_h

def save_image(im: Image.Image, out_path: Path, quality: int = 95):
    suf = out_path.suffix.lower()
    if suf in [".jpg", ".jpeg"]:
        # JPEG can't store alpha; ensure RGB
        if im.mode not in ("RGB", "L"):
            im = im.convert("RGB")
        im.save(out_path, quality=quality, subsampling=0, optimize=True)
    elif suf == ".png":
        im.save(out_path, optimize=True)
    else:
        im.save(out_path)

def resize_one(in_path: Path, out_path: Path, long_side: int = 1024, square_size: int = 1024, quality: int = 95):
    with Image.open(in_path) as im:
        w, h = im.size
        new_w, new_h = compute_new_size(w, h, long_side, square_size)

        if (new_w, new_h) != (w, h):
            im = im.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_image(im, out_path, quality=quality)
        return (w, h), (new_w, new_h)

def iter_images(in_dir: Path, recursive: bool):
    if recursive:
        for p in in_dir.rglob("*"):
            if p.is_file() and p.suffix in IMG_EXTS:
                yield p
    else:
        for p in in_dir.iterdir():
            if p.is_file() and p.suffix in IMG_EXTS:
                yield p

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", type=Path, required=True, help="Input folder containing images")
    ap.add_argument("--out_dir", type=Path, default=None, help="Output folder (ignored if --inplace)")
    ap.add_argument("--inplace", action="store_true", help="Overwrite original images in place")
    ap.add_argument("--long_side", type=int, default=1024, help="Long side size for non-square images")
    ap.add_argument("--square", type=int, default=1024, help="Exact size for square images (WxH)")
    ap.add_argument("--recursive", action="store_true", help="Process images recursively")
    ap.add_argument("--quality", type=int, default=95, help="JPEG quality")
    ap.add_argument("--overwrite", action="store_true",
                    help="When using --out_dir: overwrite existing outputs. Ignored for --inplace (always overwrites).")
    args = ap.parse_args()

    in_dir: Path = args.in_dir
    if not in_dir.is_dir():
        raise SystemExit(f"[ERROR] in_dir not found: {in_dir}")

    if args.inplace:
        out_dir = in_dir
    else:
        if args.out_dir is None:
            raise SystemExit("[ERROR] Please provide --out_dir, or use --inplace to overwrite originals.")
        out_dir = args.out_dir

    n_total, n_done, n_skip = 0, 0, 0

    for in_path in iter_images(in_dir, args.recursive):
        n_total += 1

        if args.inplace:
            # write to temp then atomically replace
            tmp_path = in_path.with_name(in_path.stem + ".tmp_resize" + in_path.suffix)
            old_sz, new_sz = resize_one(
                in_path, tmp_path,
                long_side=args.long_side,
                square_size=args.square,
                quality=args.quality
            )
            tmp_path.replace(in_path)  # atomic on same filesystem
            n_done += 1
            print(f"[OK][inplace] {in_path}  {old_sz} -> {new_sz}")
        else:
            rel = in_path.relative_to(in_dir)
            out_path = out_dir / rel

            if out_path.exists() and not args.overwrite:
                n_skip += 1
                continue

            old_sz, new_sz = resize_one(
                in_path, out_path,
                long_side=args.long_side,
                square_size=args.square,
                quality=args.quality
            )
            n_done += 1
            print(f"[OK] {in_path}  {old_sz} -> {new_sz}  -> {out_path}")

    print(f"\nDone. total={n_total}, processed={n_done}, skipped(existing)={n_skip}")
    print(f"Mode: {'inplace' if args.inplace else 'out_dir'}")

if __name__ == "__main__":
    main()
