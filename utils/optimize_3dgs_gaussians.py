#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Remove overly large Gaussians (scale outliers) from a 3DGS PLY file.

- Reads 3DGS point_cloud.ply (must contain scale_0, scale_1, scale_2).
- Computes true 3D radii (exp of log-scales).
- Optionally uses outside_ratio.npy to only prune Gaussians that are both:
    * very large
    * and often outside the alpha mask (high outside_ratio)
- Writes a cleaned PLY with the same vertex structure but fewer Gaussians.
- Optionally saves histogram & CDF plots of the radius distribution.

Intended to be used as a post-processing step in combination with the
outside_ratio-based floater removal pipeline.
"""

import os
import sys
import argparse

import numpy as np
from plyfile import PlyData, PlyElement

import matplotlib
matplotlib.use("Agg")  # Safe for headless environments
import matplotlib.pyplot as plt


def load_scales_from_ply(ply_path: str):
    """
    Load 3DGS scales (log-scale) from a PLY file.

    Returns:
        vertex: ply['vertex'] element
        scales_log: (N,3) array of log-scales
    """
    if not os.path.isfile(ply_path):
        raise FileNotFoundError(f"PLY not found: {ply_path}")

    ply = PlyData.read(ply_path)
    vertex = ply["vertex"]
    names = vertex.data.dtype.names

    required = ("scale_0", "scale_1", "scale_2")
    if not all(r in names for r in required):
        raise RuntimeError(
            f"PLY does not contain required scale fields {required}, "
            f"found: {names}"
        )

    scales_log = np.stack(
        [vertex["scale_0"], vertex["scale_1"], vertex["scale_2"]], axis=1
    ).astype(np.float32)

    return vertex, scales_log, ply


def compute_radius_from_scales(scales_log: np.ndarray, mode: str = "max"):
    """
    Compute a single scalar radius per Gaussian from log-scales.

    Args:
        scales_log: (N,3) log-scales (e.g. from 3DGS).
        mode: 'max' or 'mean'. For 'max', radius = max(exp(scale_i)).
              For 'mean', radius = mean(exp(scale_i)).

    Returns:
        radii: (N,) float array
    """
    scales = np.exp(scales_log)  # convert log-scale to true scale
    if mode == "max":
        radii = scales.max(axis=1)
    elif mode == "mean":
        radii = scales.mean(axis=1)
    else:
        raise ValueError(f"Unknown radius mode: {mode}")
    return radii


def plot_radius_distribution(radii: np.ndarray, out_dir: str, prefix: str = "radius"):
    """
    Save histogram and CDF plots of the radius distribution.

    Args:
        radii: (N,) radius values
        out_dir: directory to save PNG files
        prefix: file name prefix
    """
    os.makedirs(out_dir, exist_ok=True)

    # Histogram
    plt.figure(figsize=(6, 4))
    # Robust range: ignore extreme outliers for nicer visualization
    r_min = np.percentile(radii, 0.5)
    r_max = np.percentile(radii, 99.5)
    plt.hist(
        radii,
        bins=100,
        range=(r_min, r_max),
        density=True,
    )
    plt.xlabel("radius")
    plt.ylabel("density")
    plt.title("Histogram of Gaussian radii")
    plt.tight_layout()
    hist_path = os.path.join(out_dir, f"{prefix}_histogram.png")
    plt.savefig(hist_path)
    plt.close()
    print(f"[PLOT] Saved radius histogram to: {hist_path}")

    # CDF
    radii_sorted = np.sort(radii)
    cdf = np.linspace(0.0, 1.0, len(radii_sorted), endpoint=True)
    plt.figure(figsize=(6, 4))
    plt.plot(radii_sorted, cdf)
    plt.xlabel("radius (sorted)")
    plt.ylabel("CDF")
    plt.title("CDF of Gaussian radii")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    cdf_path = os.path.join(out_dir, f"{prefix}_cdf.png")
    plt.savefig(cdf_path)
    plt.close()
    print(f"[PLOT] Saved radius CDF to: {cdf_path}")

    # Simple text summary
    summary_path = os.path.join(out_dir, f"{prefix}_stats.txt")
    with open(summary_path, "w") as f:
        f.write("Radius statistics:\n")
        f.write(f"min:      {radii.min():.6e}\n")
        f.write(f"max:      {radii.max():.6e}\n")
        f.write(f"mean:     {radii.mean():.6e}\n")
        f.write(f"median:   {np.median(radii):.6e}\n")
        f.write(f"p90:      {np.percentile(radii, 90):.6e}\n")
        f.write(f"p95:      {np.percentile(radii, 95):.6e}\n")
        f.write(f"p99:      {np.percentile(radii, 99):.6e}\n")
        f.write(f"p99.5:    {np.percentile(radii, 99.5):.6e}\n")

    print(f"[PLOT] Saved radius stats to: {summary_path}")


def load_outside_ratio_if_available(ratio_path: str, num_vertices: int):
    """
    Load outside_ratio.npy if path is provided and file exists.

    Returns:
        ratio: (N,) array or None if not available
    """
    if ratio_path is None or ratio_path == "":
        return None

    if not os.path.isfile(ratio_path):
        print(f"[RATIO] Warning: ratio file not found: {ratio_path} (ignored)")
        return None

    ratio = np.load(ratio_path)
    if ratio.shape[0] != num_vertices:
        raise ValueError(
            f"outside_ratio length {ratio.shape[0]} != #vertices {num_vertices}"
        )

    print(f"[RATIO] Loaded outside_ratio from: {ratio_path}")
    print(
        f"[RATIO] stats: min={ratio.min():.4f}, max={ratio.max():.4f}, mean={ratio.mean():.4f}"
    )
    return ratio


def prune_big_gaussians(
    ply_path: str,
    output_ply_path: str,
    ratio_path: str = "",
    radius_mode: str = "max",
    size_threshold: float = None,
    size_percentile: float = 99.5,
    ratio_min_to_prune: float = 0.0,
    plot_dir: str = "",
):
    """
    Main pruning function.

    Args:
        ply_path: path to original 3DGS PLY.
        output_ply_path: path to save cleaned PLY.
        ratio_path: optional path to outside_ratio.npy. If provided and ratio_min_to_prune > 0,
                    only Gaussians with outside_ratio >= ratio_min_to_prune are eligible for removal.
        radius_mode: 'max' or 'mean' – how to aggregate the 3 axis scales into a scalar radius.
        size_threshold: absolute radius threshold. If None, will be computed from size_percentile.
        size_percentile: percentile used to derive size_threshold from the radius distribution
                         (e.g. 99.5 -> remove top 0.5% largest radii).
        ratio_min_to_prune: minimum outside_ratio required for pruning. E.g. 0.3 means:
                            only Gaussians with outside_ratio >= 0.3 AND radius > size_threshold
                            will be removed.
        plot_dir: optional directory to save radius histogram / CDF plots.
    """
    print(f"[INFO] Loading PLY from: {ply_path}")
    vertex, scales_log, ply = load_scales_from_ply(ply_path)
    N = len(vertex)
    print(f"[INFO] #Gaussians: {N}")

    # Compute scalar radius
    radii = compute_radius_from_scales(scales_log, mode=radius_mode)

    # Optional analysis plots
    if plot_dir:
        plot_radius_distribution(radii, plot_dir, prefix="radius")

    # Determine size threshold
    if size_threshold is None:
        size_threshold = float(np.percentile(radii, size_percentile))
        print(
            f"[THR] Using percentile-based threshold: p{size_percentile:.2f} = {size_threshold:.6e}"
        )
    else:
        print(f"[THR] Using user-specified radius threshold: {size_threshold:.6e}")

    # Load outside_ratio if provided
    ratio = load_outside_ratio_if_available(ratio_path, N)

    # Build pruning mask
    too_big = radii > size_threshold

    if ratio is not None and ratio_min_to_prune > 0.0:
        # Only prune Gaussians that are both large and often outside the mask
        cond_ratio = ratio >= ratio_min_to_prune
        bad = too_big & cond_ratio
        print(
            f"[MASK] ratio-based pruning: radius>thr AND outside_ratio>={ratio_min_to_prune}"
        )
    else:
        bad = too_big
        print(f"[MASK] pure radius-based pruning: radius>thr")

    keep = ~bad
    num_removed = int(bad.sum())
    num_kept = int(keep.sum())
    print(
        f"[RESULT] Will remove {num_removed} Gaussians (keep {num_kept}/{N}, {100.0 * num_kept / N:.2f}%)"
    )

    kept_indices = np.where(keep)[0]
    new_vertices = vertex[kept_indices]

    # Write cleaned PLY
    new_ply = PlyData(
        [PlyElement.describe(new_vertices, "vertex")],
        text=ply.text,
    )

    os.makedirs(os.path.dirname(output_ply_path), exist_ok=True)
    new_ply.write(output_ply_path)
    print(f"[SAVE] Cleaned PLY saved to: {output_ply_path}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Prune overly large Gaussians (scale outliers) from a 3DGS PLY, "
            "optionally using outside_ratio as an additional pruning condition."
        )
    )

    parser.add_argument(
        "--ply_path",
        type=str,
        required=True,
        help="Path to input 3DGS point_cloud.ply",
    )
    parser.add_argument(
        "--output_ply",
        type=str,
        required=True,
        help="Path to save cleaned PLY",
    )

    parser.add_argument(
        "--ratio_path",
        type=str,
        default="",
        help="Optional path to outside_ratio.npy (same length as #Gaussians).",
    )
    parser.add_argument(
        "--radius_mode",
        type=str,
        default="max",
        choices=["max", "mean"],
        help="How to aggregate the 3 log-scales into a scalar radius: 'max' or 'mean'.",
    )
    parser.add_argument(
        "--size_threshold",
        type=float,
        default=None,
        help=(
            "Absolute radius threshold. "
            "If not provided, will be computed from --size_percentile."
        ),
    )
    parser.add_argument(
        "--size_percentile",
        type=float,
        default=99.5,
        help=(
            "Percentile used to derive radius threshold from the distribution. "
            "E.g. 99.5 means we prune the largest 0.5%% of Gaussians by radius. "
            "Only used if --size_threshold is not set."
        ),
    )
    parser.add_argument(
        "--ratio_min_to_prune",
        type=float,
        default=0.0,
        help=(
            "If > 0 and --ratio_path is provided, only Gaussians with "
            "outside_ratio >= ratio_min_to_prune AND radius > threshold "
            "will be removed. If 0, outside_ratio is ignored in pruning."
        ),
    )
    parser.add_argument(
        "--plot_dir",
        type=str,
        default="",
        help="Optional directory to save radius histogram / CDF plots.",
    )

    args = parser.parse_args(sys.argv[1:])

    prune_big_gaussians(
        ply_path=args.ply_path,
        output_ply_path=args.output_ply,
        ratio_path=args.ratio_path,
        radius_mode=args.radius_mode,
        size_threshold=args.size_threshold,
        size_percentile=args.size_percentile,
        ratio_min_to_prune=args.ratio_min_to_prune,
        plot_dir=args.plot_dir,
    )


if __name__ == "__main__":
    main()
