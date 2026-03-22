#!/usr/bin/env python3
import argparse
from pathlib import Path
import csv

import numpy as np
import cv2


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root_birefnet", type=str, required=True, help="e.g. /path/to/root_birefnet")
    p.add_argument("--classes", nargs="*", default=None, help="Optional class list (order matters). If omitted, auto-detect.")
    p.add_argument("--splits", nargs="*", default=["train_msk", "test_msk"], help="Mask dirs to check.")
    p.add_argument("--ext", type=str, default=".png")
    p.add_argument("--min_area", type=int, default=50, help="Ignore tiny components smaller than this (pixels).")
    p.add_argument("--fix", action="store_true", help="Keep only the largest component and overwrite (or save to --fix_out_subdir).")
    p.add_argument("--fix_out_subdir", type=str, default="", help="If set, save fixed masks under this subdir instead of overwriting.")
    p.add_argument("--report_csv", type=str, default="mask_quality_report.csv", help="CSV filename (saved under root_birefnet).")
    p.add_argument("--break_after_one", action="store_true", help="Debug: process only first class.")
    return p.parse_args()


def binarize_mask(path: Path) -> np.ndarray:
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(f"Failed to read: {path}")
    # treat >=128 as foreground
    return (m >= 128).astype(np.uint8)


def analyze_components(mask01: np.ndarray, min_area: int):
    """
    Returns:
      n_kept: number of components with area>=min_area
      areas: list of kept component areas (sorted desc)
      labels: connected component label map (0=bg)
      stats: stats from cv2.connectedComponentsWithStats
      kept_ids: component ids kept after min_area filter (excluding bg=0)
    """
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask01, connectivity=8)
    # stats: [id, x, y, w, h, area], id=0 is background
    kept = []
    for cid in range(1, num):
        area = int(stats[cid, cv2.CC_STAT_AREA])
        if area >= min_area:
            kept.append((cid, area))
    kept.sort(key=lambda x: x[1], reverse=True)
    kept_ids = [cid for cid, _ in kept]
    areas = [area for _, area in kept]
    return len(kept_ids), areas, labels, stats, kept_ids


def keep_largest_component(mask01: np.ndarray, labels: np.ndarray, kept_ids: list[int]) -> np.ndarray:
    if not kept_ids:
        return np.zeros_like(mask01, dtype=np.uint8)
    largest = kept_ids[0]
    out = (labels == largest).astype(np.uint8)
    return out


def ensure_output_mask(mask01: np.ndarray) -> np.ndarray:
    # normalize to 0/255 for saving
    return (mask01.astype(np.uint8) * 255)


def main():
    args = parse_args()
    rb = Path(args.root_birefnet)

    if args.classes is None:
        classes = sorted([p.name for p in rb.iterdir() if p.is_dir()])
    else:
        classes = args.classes

    report_path = rb / args.report_csv
    rows = []

    total_bad = 0
    total_seen = 0

    for ci, cls in enumerate(classes):
        cls_dir = rb / cls
        if not cls_dir.is_dir():
            print(f"[warn] missing class dir: {cls_dir}")
            continue

        cls_seen = 0
        cls_bad = 0
        cls_multi = 0
        cls_empty = 0

        for split in args.splits:
            split_dir = cls_dir / split
            if not split_dir.is_dir():
                continue

            for mpath in split_dir.rglob(f"*{args.ext}"):
                cls_seen += 1
                total_seen += 1

                rel = mpath.relative_to(cls_dir)
                mask01 = binarize_mask(mpath)

                n_comp, areas, labels, stats, kept_ids = analyze_components(mask01, args.min_area)
                fg_area = int(mask01.sum())
                largest_area = int(areas[0]) if areas else 0
                largest_ratio = (largest_area / fg_area) if fg_area > 0 else 0.0

                ok_single = (n_comp == 1)
                is_empty = (fg_area == 0)

                if is_empty:
                    cls_empty += 1
                    cls_bad += 1
                    total_bad += 1
                elif not ok_single:
                    cls_multi += 1
                    cls_bad += 1
                    total_bad += 1

                # optional fix
                fixed_path = ""
                if args.fix:
                    fixed01 = keep_largest_component(mask01, labels, kept_ids)
                    if args.fix_out_subdir:
                        out_dir = cls_dir / args.fix_out_subdir / rel.parent
                        out_dir.mkdir(parents=True, exist_ok=True)
                        out_path = out_dir / rel.name
                    else:
                        out_path = mpath  # overwrite
                    cv2.imwrite(str(out_path), ensure_output_mask(fixed01))
                    fixed_path = str(out_path.relative_to(rb))

                rows.append({
                    "class": cls,
                    "split": split,
                    "rel_path": str(rel),
                    "n_components_ge_min_area": n_comp,
                    "fg_area_px": fg_area,
                    "largest_area_px": largest_area,
                    "largest_ratio": f"{largest_ratio:.6f}",
                    "is_empty": int(is_empty),
                    "ok_single_component": int(ok_single and not is_empty),
                    "fixed_path": fixed_path,
                })

        if cls_seen:
            print("============================================================")
            print(f"[class] {cls}")
            print(f"  seen: {cls_seen}")
            print(f"  bad : {cls_bad}  (empty: {cls_empty}, multi-component: {cls_multi})")
            if cls_seen > 0:
                print(f"  bad_rate: {cls_bad/cls_seen:.3%}")
        else:
            print(f"[class] {cls}  seen: 0 (no masks found in {args.splits})")

        if args.break_after_one and ci == 0:
            print("[info] break_after_one enabled -> stop after first class.")
            break

    # write csv
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", newline="") as f:
        fieldnames = list(rows[0].keys()) if rows else [
            "class","split","rel_path","n_components_ge_min_area","fg_area_px",
            "largest_area_px","largest_ratio","is_empty","ok_single_component","fixed_path"
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print("------------------------------------------------------------")
    print(f"[all] seen={total_seen}  bad={total_bad}  bad_rate={(total_bad/total_seen if total_seen else 0):.3%}")
    print(f"[csv] {report_path}")


if __name__ == "__main__":
    main()