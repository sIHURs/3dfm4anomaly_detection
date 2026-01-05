#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import cv2
import numpy as np
from sklearn.cluster import KMeans
from pathlib import Path
from tqdm import tqdm   # ← 新增


def remove_bg_kmeans(
    img: np.ndarray,
    erode_iter: int = 1,
) -> np.ndarray:
    h, w = img.shape[:2]

    # --- KMeans segmentation ---
    X = img.reshape(-1, 3).astype(np.float32)
    kmeans = KMeans(n_clusters=2, n_init="auto", random_state=0).fit(X)
    labels = kmeans.labels_.reshape(h, w)
    centers = kmeans.cluster_centers_

    # --- Identify background cluster (closest to white or black) ---
    white = np.array([255, 255, 255], np.float32)
    black = np.array([0, 0, 0], np.float32)

    d_to_white = np.linalg.norm(centers - white, axis=1)
    d_to_black = np.linalg.norm(centers - black, axis=1)
    bg_idx = int(np.argmin(np.minimum(d_to_white, d_to_black)))

    mask = (labels != bg_idx).astype(np.uint8) * 255

    if erode_iter > 0:
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=erode_iter)

    rgba = np.dstack([img, mask])
    return rgba


def process_dir(
    in_dir: Path,
    out_dir: Path,
    erode_iter: int,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    exts = {".png", ".jpg", ".jpeg"}

    img_paths = [
        p for p in sorted(in_dir.iterdir())
        if p.suffix.lower() in exts
    ]

    for img_path in tqdm(
        img_paths,
        desc=f"Processing {in_dir.name}",
        unit="img"
    ):
        img = cv2.imread(str(img_path))
        if img is None:
            tqdm.write(f"[WARN] Failed to read {img_path}")
            continue

        rgba = remove_bg_kmeans(img, erode_iter=erode_iter)

        out_path = out_dir / f"{img_path.stem}_rgba.png"
        cv2.imwrite(str(out_path), rgba)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Remove solid-color background (white/black) using KMeans and output RGBA images."
    )
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--erode_iter", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    process_dir(
        in_dir=Path(args.input_dir),
        out_dir=Path(args.output_dir),
        erode_iter=args.erode_iter,
    )
