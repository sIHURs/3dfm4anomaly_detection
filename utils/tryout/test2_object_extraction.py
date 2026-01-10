#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def process_image(img: np.ndarray, thr: int, erode_iter: int) -> np.ndarray:
    g = img[:, :, 1]
    binary = cv2.threshold(g, thr, 1, cv2.THRESH_BINARY)[1]  # 0/1
    mask01 = np.abs(binary - 1).astype(np.uint8)            # invert -> 0/1
    mask = (mask01 * 255).astype(np.uint8)                  # 0/255

    if erode_iter > 0:
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=erode_iter)

    rgba = np.dstack([img, mask])  # BGRA for OpenCV
    return rgba

def remove_white_background_bgr(img, erode_iter=1, thresh=254):
    """
    img: BGR image (OpenCV)
    return: RGBA image with white background removed
    """
    # use green channel
    g = img[:, :, 1]

    # background ≈ white
    _, bg_mask = cv2.threshold(g, thresh, 1, cv2.THRESH_BINARY)

    if erode_iter > 0:
        kernel = np.ones((3, 3), np.uint8)
        bg_mask = cv2.dilate(bg_mask, kernel, iterations=erode_iter)

    # invert: foreground=1, background=0
    alpha = np.abs(bg_mask - 1).astype(np.uint8)

    rgba = np.dstack((img, alpha))
    return rgba


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", required=True, type=str)
    p.add_argument("--output_dir", required=True, type=str)
    p.add_argument("--thr", type=int, default=254)
    p.add_argument("--erode_iter", type=int, default=1)
    p.add_argument("--exts", type=str, default="png,jpg,jpeg")
    args = p.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exts = {("." + e.strip().lower().lstrip(".")) for e in args.exts.split(",") if e.strip()}
    paths = [p for p in sorted(in_dir.iterdir()) if p.is_file() and p.suffix.lower() in exts]

    for img_path in tqdm(paths, desc=f"Processing {in_dir.name}", unit="img"):
        img = cv2.imread(str(img_path))
        if img is None:
            tqdm.write(f"[WARN] read failed: {img_path}")
            continue

        # rgba = process_image(img, thr=args.thr, erode_iter=args.erode_iter)
        rgba = remove_white_background_bgr(img)
        out_path = out_dir / f"{img_path.stem}.png"  # force PNG to keep alpha
        cv2.imwrite(str(out_path), rgba)

    print(f"[OK] Done. Saved to: {out_dir}")


if __name__ == "__main__":
    main()
