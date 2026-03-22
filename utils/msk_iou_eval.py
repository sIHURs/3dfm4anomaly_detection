#!/usr/bin/env python3
"""
Per-class IoU evaluation + outlier analysis + visualization.

Compute IoU between:
(A) mask derived from white-background composited images in root
    (foreground = not-white pixels)
and
(B) BiRefNet masks saved in root_birefnet/{train_msk,test_msk}/...

This evaluates how well BiRefNet masks agree with the reference
foreground regions encoded in the white-background images under root.

Extra features:
- per-class descriptive statistics
- IQR-based or percentile-based outlier detection
- outlier comparison figures
- CSV export of all results / outliers
- box plots for mask quality inspection

Example:
  python eval_iou_from_whitebg.py \
    --root /data/Anomaly_refine_msk \
    --root_birefnet /home/wangyifa/root_birefnet \
    --classes rubberduck binderclip2 \
    --tol 5 \
    --save_white_msk \
    --save_plots
"""

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--root",
        type=str,
        required=True,
        help="Reference dataset root containing white-background images used to derive mask A.",
    )
    p.add_argument(
        "--root_birefnet",
        type=str,
        required=True,
        help="BiRefNet dataset root containing white-bg images + train_msk/test_msk.",
    )
    p.add_argument(
        "--classes",
        nargs="*",
        default=None,
        help="Class list (order matters). If omitted, auto-detect from --root.",
    )
    p.add_argument("--ext", type=str, default=".png", help="Image extension, default .png")
    p.add_argument(
        "--tol",
        type=int,
        default=5,
        help="White tolerance (0..255). Near-white if RGB >= 255 - tol.",
    )
    p.add_argument(
        "--skip_missing",
        action="store_true",
        help="Skip if any file missing (otherwise raise).",
    )

    p.add_argument(
        "--save_white_msk",
        action="store_true",
        help="Also save masks derived from reference white-bg images for debugging.",
    )
    p.add_argument(
        "--white_msk_suffix",
        type=str,
        default="",
        help="Optional suffix for saved masks, e.g. '_white'. Default '' keeps same name.",
    )

    p.add_argument(
        "--analysis_dir",
        type=str,
        default=None,
        help="Output directory for analysis files. Default: <root_birefnet>/_iou_analysis",
    )
    p.add_argument(
        "--save_plots",
        action="store_true",
        help="Save boxplots and outlier visualizations.",
    )
    p.add_argument(
        "--outlier_method",
        type=str,
        default="iqr",
        choices=["iqr", "percentile"],
        help="Outlier detection method.",
    )
    p.add_argument(
        "--outlier_iqr_k",
        type=float,
        default=1.5,
        help="IQR multiplier for outlier detection.",
    )
    p.add_argument(
        "--outlier_pct",
        type=float,
        default=5.0,
        help="Percentile threshold if outlier_method=percentile. Lower tail only.",
    )
    p.add_argument(
        "--max_outlier_plots_per_class",
        type=int,
        default=30,
        help="Maximum number of outlier comparison figures to save per class.",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=160,
        help="DPI for saved figures.",
    )

    p.add_argument(
        "--break_after_one",
        action="store_true",
        help="Process only the first class then exit (debug).",
    )
    return p.parse_args()


def load_mask_01(mask_path: Path) -> np.ndarray:
    m = np.array(Image.open(mask_path).convert("L"))
    return (m >= 128).astype(np.uint8)


def load_rgb(img_path: Path) -> np.ndarray:
    return np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)


def mask_from_whitebg(img_path: Path, tol: int) -> np.ndarray:
    """
    Foreground = pixel is NOT near-white.
    near-white condition: R,G,B >= 255 - tol
    """
    rgb = load_rgb(img_path)
    thr = 255 - tol
    near_white = (rgb[..., 0] >= thr) & (rgb[..., 1] >= thr) & (rgb[..., 2] >= thr)
    return (~near_white).astype(np.uint8)


def save_mask01(mask01: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask01 * 255).astype(np.uint8)).save(path)


def compute_iou(a01: np.ndarray, b01: np.ndarray) -> float:
    if a01.shape != b01.shape:
        raise ValueError(f"Shape mismatch: {a01.shape} vs {b01.shape}")
    a = a01.astype(bool)
    b = b01.astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0
    return float(inter / union)


def iter_image_paths(root_cls: Path, ext: str):
    for split in ("train", "test"):
        split_dir = root_cls / split
        if not split_dir.is_dir():
            continue
        yield from split_dir.rglob(f"*{ext}")


def get_expected_mask_path(root_birefnet_cls: Path, rel_under_cls: Path) -> Path:
    """
    rel_under_cls: train/good/xxx.png or test/scratch/xxx.png
    -> root_birefnet/<class>/train_msk/good/xxx.png
    -> root_birefnet/<class>/test_msk/scratch/xxx.png
    """
    parts = rel_under_cls.parts
    if len(parts) < 2:
        raise ValueError(f"Unexpected rel path: {rel_under_cls}")

    split = parts[0]
    rest = Path(*parts[1:])

    if split == "train":
        return root_birefnet_cls / "train_msk" / rest
    if split == "test":
        return root_birefnet_cls / "test_msk" / rest
    raise ValueError(f"Unexpected split in path: {rel_under_cls}")


def maybe_save_white_mask(
    *,
    args,
    analysis_root: Path,
    cls: str,
    rel_under_cls: Path,
    a01: np.ndarray,
):
    if not args.save_white_msk:
        return

    parts = rel_under_cls.parts
    split = parts[0]
    rest = Path(*parts[1:])

    stem = rest.stem
    ext = rest.suffix
    out_name = f"{stem}{args.white_msk_suffix}{ext}"

    if split == "train":
        out_path = analysis_root / "saved_ref_masks" / cls / "train_msk_refwhite" / rest.parent / out_name
    else:
        out_path = analysis_root / "saved_ref_masks" / cls / "test_msk_refwhite" / rest.parent / out_name

    save_mask01(a01, out_path)


def resize_mask_to(mask01: np.ndarray, target_hw) -> np.ndarray:
    h, w = target_hw
    if mask01.shape == (h, w):
        return mask01
    img = Image.fromarray((mask01 * 255).astype(np.uint8))
    img = img.resize((w, h), resample=Image.NEAREST)
    return (np.array(img) >= 128).astype(np.uint8)


def compute_confusion_map(a01: np.ndarray, b01: np.ndarray) -> np.ndarray:
    """
    RGB visualization:
      TP: green
      FP: red   (A=1, B=0)
      FN: blue  (A=0, B=1)
      TN: white
    """
    a = a01.astype(bool)
    b = b01.astype(bool)

    out = np.ones((a.shape[0], a.shape[1], 3), dtype=np.uint8) * 255
    tp = a & b
    fp = a & (~b)
    fn = (~a) & b

    out[tp] = np.array([0, 180, 0], dtype=np.uint8)
    out[fp] = np.array([220, 30, 30], dtype=np.uint8)
    out[fn] = np.array([30, 80, 220], dtype=np.uint8)
    return out


def summarize(values: list[float]) -> dict:
    if not values:
        return {
            "n": 0,
            "mean": math.nan,
            "median": math.nan,
            "std": math.nan,
            "min": math.nan,
            "max": math.nan,
            "q1": math.nan,
            "q3": math.nan,
            "iqr": math.nan,
        }

    arr = np.asarray(values, dtype=np.float64)
    q1 = float(np.percentile(arr, 25))
    q3 = float(np.percentile(arr, 75))
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "q1": q1,
        "q3": q3,
        "iqr": float(q3 - q1),
    }


def detect_outliers(
    records: list[dict],
    method: str,
    iqr_k: float,
    pct: float,
) -> tuple[list[dict], dict]:
    if not records:
        return [], {}

    vals = np.array([r["iou"] for r in records], dtype=np.float64)

    if method == "iqr":
        q1 = float(np.percentile(vals, 25))
        q3 = float(np.percentile(vals, 75))
        iqr = q3 - q1
        lower = q1 - iqr_k * iqr
        outliers = [r for r in records if r["iou"] < lower]
        info = {"method": "iqr", "q1": q1, "q3": q3, "iqr": iqr, "lower": lower}
        return outliers, info

    if method == "percentile":
        thr = float(np.percentile(vals, pct))
        outliers = [r for r in records if r["iou"] <= thr]
        info = {"method": "percentile", "pct": pct, "lower": thr}
        return outliers, info

    raise ValueError(f"Unknown outlier method: {method}")


def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]):
    ensure_parent(path)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_outlier_figure(record: dict, out_path: Path, dpi: int = 160):
    ref_img_path = Path(record["ref_img_path"])
    birefnet_img_path = Path(record["masked_img_path"])
    mask_path = Path(record["mask_path"])

    a01 = (
        load_mask_01(Path(record["derived_mask_tmp_path"]))
        if record.get("derived_mask_tmp_path")
        else None
    )

    ref_whitebg_rgb = load_rgb(ref_img_path)
    birefnet_whitebg_rgb = load_rgb(birefnet_img_path)

    if a01 is None:
        a01 = mask_from_whitebg(ref_img_path, tol=record["tol"])

    b01 = load_mask_01(mask_path)
    b01 = resize_mask_to(b01, a01.shape)

    diff = compute_confusion_map(a01, b01)

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    axes[0].imshow(ref_whitebg_rgb)
    axes[0].set_title("Reference white-bg")

    axes[1].imshow(birefnet_whitebg_rgb)
    axes[1].set_title("BiRefNet white-bg")

    axes[2].imshow(a01, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("Reference mask (A)")

    axes[3].imshow(b01, cmap="gray", vmin=0, vmax=1)
    axes[3].set_title("BiRefNet mask (B)")

    axes[4].imshow(diff)
    axes[4].set_title("Diff: TP=green FP=red FN=blue")

    for ax in axes:
        ax.axis("off")

    rel_text = record["rel_path"].replace("\\", "/")
    fig.suptitle(
        f'class={record["class"]} | IoU={record["iou"]:.4f} | {rel_text}',
        fontsize=11,
    )
    fig.tight_layout()

    ensure_parent(out_path)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_boxplot_per_class(
    class_to_ious: dict[str, list[float]],
    out_path: Path,
    dpi: int = 160,
):
    valid_items = [(cls, vals) for cls, vals in class_to_ious.items() if len(vals) > 0]
    if not valid_items:
        return

    labels = [k for k, _ in valid_items]
    data = [v for _, v in valid_items]

    fig_w = max(8, 0.45 * len(labels) + 4)
    fig, ax = plt.subplots(figsize=(fig_w, 6))
    ax.boxplot(data, tick_labels=labels, showfliers=True)
    ax.set_ylabel("IoU")
    ax.set_title("Per-class IoU boxplot")
    ax.set_ylim(0.0, 1.02)
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()

    ensure_parent(out_path)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_single_class_boxplot(cls: str, ious: list[float], out_path: Path, dpi: int = 160):
    if not ious:
        return

    fig, ax = plt.subplots(figsize=(4.5, 6))
    ax.boxplot([ious], tick_labels=[cls], showfliers=True)
    ax.set_ylabel("IoU")
    ax.set_title(f"IoU boxplot: {cls}")
    ax.set_ylim(0.0, 1.02)
    fig.tight_layout()

    ensure_parent(out_path)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def run_one_class(
    args,
    cls: str,
    analysis_root: Path,
) -> tuple[list[float], list[dict], list[dict], dict]:
    root = Path(args.root)
    rb = Path(args.root_birefnet)

    root_cls = root / cls
    rb_cls = rb / cls

    if not root_cls.is_dir():
        msg = f"[warn] missing class in root: {root_cls}"
        if args.skip_missing:
            print(msg)
            return [], [], [], {}
        raise FileNotFoundError(msg)

    if not rb_cls.is_dir():
        msg = f"[warn] missing class in root_birefnet: {rb_cls}"
        if args.skip_missing:
            print(msg)
            return [], [], [], {}
        raise FileNotFoundError(msg)

    imgs = list(iter_image_paths(root_cls, args.ext))
    if not imgs:
        print(f"[warn] no images found for class {cls} under train/test")
        return [], [], [], {}

    cls_ious: list[float] = []
    records: list[dict] = []

    for ref_img_path in imgs:
        rel_under_cls = ref_img_path.relative_to(root_cls)
        birefnet_whitebg_path = rb_cls / rel_under_cls
        msk_path = get_expected_mask_path(rb_cls, rel_under_cls)

        if (not birefnet_whitebg_path.exists()) or (not msk_path.exists()):
            if args.skip_missing:
                continue
            raise FileNotFoundError(
                f"[miss] {cls}/{rel_under_cls} -> "
                f"birefnet_whitebg:{birefnet_whitebg_path.exists()} "
                f"mask:{msk_path.exists()}"
            )

        # IMPORTANT:
        # Derive reference mask from white-background image in root.
        # Compare it against the BiRefNet mask under root_birefnet.
        a01 = mask_from_whitebg(ref_img_path, tol=args.tol)
        maybe_save_white_mask(
            args=args,
            analysis_root=analysis_root,
            cls=cls,
            rel_under_cls=rel_under_cls,
            a01=a01,
        )

        b01 = load_mask_01(msk_path)

        if a01.shape != b01.shape:
            b01 = resize_mask_to(b01, a01.shape)

        iou = compute_iou(a01, b01)
        cls_ious.append(iou)

        tmp_mask_path = analysis_root / "tmp_derived_masks" / cls / rel_under_cls
        save_mask01(a01, tmp_mask_path)

        records.append(
            {
                "class": cls,
                "rel_path": str(rel_under_cls),
                "split": rel_under_cls.parts[0] if len(rel_under_cls.parts) > 0 else "",
                "subdir": str(Path(*rel_under_cls.parts[1:-1])) if len(rel_under_cls.parts) > 2 else "",
                "filename": rel_under_cls.name,
                "iou": iou,
                "ref_img_path": str(ref_img_path),
                "masked_img_path": str(birefnet_whitebg_path),
                "mask_path": str(msk_path),
                "derived_mask_tmp_path": str(tmp_mask_path),
                "tol": args.tol,
            }
        )

    outliers, outlier_info = detect_outliers(
        records,
        method=args.outlier_method,
        iqr_k=args.outlier_iqr_k,
        pct=args.outlier_pct,
    )
    outliers = sorted(outliers, key=lambda x: x["iou"])

    return cls_ious, records, outliers, outlier_info


def main():
    args = parse_args()
    root = Path(args.root)
    analysis_root = Path(args.analysis_dir) if args.analysis_dir else Path(args.root_birefnet) / "_iou_analysis"
    analysis_root.mkdir(parents=True, exist_ok=True)

    if args.classes is None:
        classes = sorted([p.name for p in root.iterdir() if p.is_dir()])
    else:
        classes = args.classes

    common_fieldnames = [
        "class",
        "rel_path",
        "split",
        "subdir",
        "filename",
        "iou",
        "ref_img_path",
        "masked_img_path",
        "mask_path",
        "derived_mask_tmp_path",
        "tol",
    ]

    all_ious: list[float] = []
    all_records: list[dict] = []
    summary_rows: list[dict] = []
    class_to_ious: dict[str, list[float]] = {}

    for i, cls in enumerate(classes):
        print("============================================================")
        print(f"[info] class: {cls}")

        cls_ious, records, outliers, outlier_info = run_one_class(args, cls, analysis_root)
        class_to_ious[cls] = cls_ious
        all_ious.extend(cls_ious)
        all_records.extend(records)

        stats = summarize(cls_ious)
        stats_row = {
            "class": cls,
            **stats,
            "num_outliers": len(outliers),
        }

        if outlier_info:
            for k, v in outlier_info.items():
                stats_row[f"outlier_{k}"] = v

        summary_rows.append(stats_row)

        if cls_ious:
            print(
                f"[class] {cls:15s}  n={len(cls_ious):5d}  "
                f"mean_iou={float(np.mean(cls_ious)):.4f}  "
                f"median={float(np.median(cls_ious)):.4f}  "
                f"min={float(np.min(cls_ious)):.4f}  outliers={len(outliers)}"
            )
        else:
            print(f"[class] {cls:15s}  n=0 (no matched pairs)")

        cls_dir = analysis_root / "per_class" / cls
        cls_dir.mkdir(parents=True, exist_ok=True)

        write_csv(
            cls_dir / "all_samples.csv",
            rows=records,
            fieldnames=common_fieldnames,
        )

        write_csv(
            cls_dir / "outliers.csv",
            rows=outliers,
            fieldnames=common_fieldnames,
        )

        write_csv(
            cls_dir / "summary.csv",
            rows=[stats_row],
            fieldnames=list(stats_row.keys()),
        )

        if args.save_plots:
            save_single_class_boxplot(
                cls=cls,
                ious=cls_ious,
                out_path=cls_dir / "boxplot.png",
                dpi=args.dpi,
            )

            for j, rec in enumerate(outliers[:args.max_outlier_plots_per_class]):
                stem = Path(rec["rel_path"]).with_suffix("")
                safe_name = str(stem).replace("\\", "__").replace("/", "__")
                out_fig = cls_dir / "outliers_vis" / f"{j:03d}_iou_{rec['iou']:.4f}_{safe_name}.png"
                save_outlier_figure(rec, out_fig, dpi=args.dpi)

        if args.break_after_one and i == 0:
            print("[info] break_after_one enabled -> stop after first class.")
            break

    print("------------------------------------------------------------")
    if all_ious:
        print(
            f"[all] n={len(all_ious)}  "
            f"mean_iou={float(np.mean(all_ious)):.4f}  "
            f"median={float(np.median(all_ious)):.4f}"
        )
    else:
        print("[all] n=0 (no matched pairs)")

    write_csv(
        analysis_root / "all_samples.csv",
        rows=all_records,
        fieldnames=common_fieldnames,
    )

    write_csv(
        analysis_root / "summary_per_class.csv",
        rows=summary_rows,
        fieldnames=sorted({k for row in summary_rows for k in row.keys()}) if summary_rows else ["class"],
    )

    global_stats = summarize(all_ious)
    global_row = {"class": "__all__", **global_stats}
    write_csv(
        analysis_root / "summary_global.csv",
        rows=[global_row],
        fieldnames=list(global_row.keys()),
    )

    if args.save_plots:
        save_boxplot_per_class(
            class_to_ious=class_to_ious,
            out_path=analysis_root / "boxplot_per_class.png",
            dpi=args.dpi,
        )

        if all_ious:
            fig, ax = plt.subplots(figsize=(4.5, 6))
            ax.boxplot([all_ious], tick_labels=["all"], showfliers=True)
            ax.set_ylabel("IoU")
            ax.set_title("Global IoU boxplot")
            ax.set_ylim(0.0, 1.02)
            fig.tight_layout()
            fig.savefig(analysis_root / "boxplot_global.png", dpi=args.dpi, bbox_inches="tight")
            plt.close(fig)

    print(f"[done] analysis saved to: {analysis_root}")


if __name__ == "__main__":
    main()