#!/usr/bin/env python3
"""
Apply binary masks (train_msk/test_msk) to images (train/test) inside root_birefnet.

- For each image:
    image_out = image where mask==0 set to white (255,255,255)
- Assumes masks have the same relative path under *_msk as images under split.

Example:
  python apply_msk_whitebg.py \
    --root_birefnet /path/to/root_birefnet \
    --classes rubberduck binderclip2 \
    --out_train_dir train_masked \
    --out_test_dir test_masked \
    --overwrite

To overwrite images in-place:
  python apply_msk_whitebg.py \
    --root_birefnet /path/to/root_birefnet \
    --inplace
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root_birefnet", type=str, required=True)
    p.add_argument("--classes", nargs="*", default=None, help="If omitted, auto-detect class dirs under root_birefnet.")
    p.add_argument("--img_ext", type=str, default=".png", help="Image extension, default .png")
    p.add_argument("--mask_ext", type=str, default=".png", help="Mask extension, default .png")

    p.add_argument("--train_dir", type=str, default="train", help="Image train dir name")
    p.add_argument("--test_dir", type=str, default="test", help="Image test dir name")
    p.add_argument("--train_msk_dir", type=str, default="train_msk", help="Mask train dir name")
    p.add_argument("--test_msk_dir", type=str, default="test_msk", help="Mask test dir name")

    p.add_argument("--out_train_dir", type=str, default="train_masked", help="Output dir for train masked images")
    p.add_argument("--out_test_dir", type=str, default="test_masked", help="Output dir for test masked images")

    p.add_argument("--inplace", action="store_true", help="Overwrite train/test images directly (DANGEROUS).")
    p.add_argument("--overwrite", action="store_true", help="Overwrite outputs if exist.")
    p.add_argument("--skip_missing", action="store_true", help="Skip if mask missing (otherwise raise).")

    p.add_argument("--mask_threshold", type=int, default=128, help=">= threshold treated as foreground (1).")
    return p.parse_args()


def load_image_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)


def load_mask01(path: Path, thr: int) -> np.ndarray:
    m = np.array(Image.open(path).convert("L"))
    return (m >= thr).astype(np.uint8)


def apply_white_bg(img_rgb: np.ndarray, mask01: np.ndarray) -> np.ndarray:
    if mask01.shape != img_rgb.shape[:2]:
        # resize mask to image size (nearest)
        m = Image.fromarray((mask01 * 255).astype(np.uint8)).resize(
            (img_rgb.shape[1], img_rgb.shape[0]), resample=Image.NEAREST
        )
        mask01 = (np.array(m) >= 128).astype(np.uint8)

    out = img_rgb.copy()
    out[mask01 == 0] = 255
    return out


def iter_split_images(cls_dir: Path, split_name: str, ext: str):
    split_dir = cls_dir / split_name
    if not split_dir.is_dir():
        return
    yield from split_dir.rglob(f"*{ext}")


def main():
    args = parse_args()
    rb = Path(args.root_birefnet)

    if args.classes is None:
        classes = sorted([p.name for p in rb.iterdir() if p.is_dir()])
    else:
        classes = args.classes

    total = 0
    ok = 0
    skipped = 0

    for cls in classes:
        cls_dir = rb / cls
        if not cls_dir.is_dir():
            print(f"[warn] missing class dir: {cls_dir}")
            continue

        print("============================================================")
        print(f"[info] class: {cls}")

        for split_name, msk_dirname, out_dirname in [
            (args.train_dir, args.train_msk_dir, args.out_train_dir),
            (args.test_dir,  args.test_msk_dir,  args.out_test_dir),
        ]:
            split_dir = cls_dir / split_name
            msk_dir = cls_dir / msk_dirname

            if not split_dir.is_dir():
                continue
            if not msk_dir.is_dir():
                msg = f"[warn] missing mask dir: {msk_dir}"
                if args.skip_missing:
                    print(msg)
                    continue
                raise FileNotFoundError(msg)

            for img_path in iter_split_images(cls_dir, split_name, args.img_ext):
                total += 1
                rel_under_split = img_path.relative_to(split_dir)  # e.g. good/001.png or scratch/001.png

                # expected mask path: <class>/<train_msk or test_msk>/<same rel path>
                mask_path = msk_dir / rel_under_split
                if mask_path.suffix != args.mask_ext:
                    mask_path = mask_path.with_suffix(args.mask_ext)

                if not mask_path.exists():
                    if args.skip_missing:
                        skipped += 1
                        continue
                    raise FileNotFoundError(f"[miss] mask not found: {mask_path} (for image {img_path})")

                # output path
                if args.inplace:
                    out_path = img_path
                else:
                    out_root = cls_dir / out_dirname
                    out_path = out_root / rel_under_split
                    out_path.parent.mkdir(parents=True, exist_ok=True)

                if (not args.overwrite) and out_path.exists() and (not args.inplace):
                    skipped += 1
                    continue

                try:
                    img = load_image_rgb(img_path)
                    m01 = load_mask01(mask_path, args.mask_threshold)
                    out = apply_white_bg(img, m01)

                    Image.fromarray(out).save(out_path)
                    ok += 1
                except Exception as e:
                    print(f"[err] {cls}/{split_name}/{rel_under_split}: {e}")

            print(f"[info] split {split_name}: done")

        print(f"[info] class {cls}: done")

    print("------------------------------------------------------------")
    print(f"[sum] total={total} ok={ok} skipped={skipped}")


if __name__ == "__main__":
    main()