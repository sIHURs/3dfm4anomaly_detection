import os
import sys
import argparse
import shutil

import cv2
import numpy as np
import torch
from plyfile import PlyData, PlyElement
from tqdm import tqdm
import colmap

# For analysis & plots
import matplotlib
matplotlib.use("Agg")  # headless 环境安全
import matplotlib.pyplot as plt

# 3DGS imports
from gaussian_splatting.arguments import ModelParams, PipelineParams
from gaussian_splatting.utils.general_utils import safe_state
from gaussian_splatting.scene import Scene, GaussianModel
from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.loss_utils import l1_loss, ssim
from gaussian_splatting.utils.image_utils import psnr

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except Exception:
    FUSED_SSIM_AVAILABLE = False


# ======================
#  COLMAP utils
# ======================

def load_colmap_model(colmap_dir: str):
    """
    Load COLMAP cameras, images and points3D from a sparse model directory.
    """
    cameras, images, points3D = colmap.read_model(colmap_dir, ext=".bin")
    return cameras, images, points3D


# ======================
#  Step 1: compute outside_ratio
# ======================

@torch.no_grad()
def compute_outside_ratio_torch(
    pts_np: np.ndarray,
    cameras,
    images,
    mask_cache_np: dict,
    device: str = "cuda",
) -> np.ndarray:
    """
    Compute outside_ratio for each 3D point using alpha masks.

    outside_ratio[i] = (# projections of point i landing on mask == 0)
                       / (# valid projections of point i)

    Args:
        pts_np:        (N,3) xyz of points (Gaussian centers).
        cameras:       COLMAP cameras dict.
        images:        COLMAP images dict.
        mask_cache_np: dict[image_id] -> (H,W) uint8 binary mask (0 or 255)
        device:        'cuda' or 'cpu'

    Returns:
        ratio_np: (N,) float32 array, outside_ratio per point.
                  Points with total_projections == 0 will have ratio = 0.
    """
    images_list = list(images.values())
    N = pts_np.shape[0]

    pts = torch.from_numpy(pts_np).to(device=device, dtype=torch.float32)  # (N,3)

    total = torch.zeros(N, device=device, dtype=torch.int32)
    outside = torch.zeros(N, device=device, dtype=torch.int32)

    # move masks to device
    mask_cache = {
        img_id: torch.from_numpy(mask_np).to(device=device)
        for img_id, mask_np in mask_cache_np.items()
    }

    print(f"[Torch] Computing outside_ratio for {N} points on device: {device}")
    for img in tqdm(images_list, desc="Images (Torch)"):
        cam = cameras[img.camera_id]
        mask = mask_cache[img.id]  # (H,W) on device

        H, W = mask.shape

        # R, t
        R_np = colmap.qvec2rotmat(img.qvec)  # (3,3)
        t_np = img.tvec.reshape(3, 1)        # (3,1)

        R = torch.from_numpy(R_np).to(device=device, dtype=torch.float32)  # (3,3)
        t = torch.from_numpy(t_np).to(device=device, dtype=torch.float32)  # (3,1)

        fx, fy, cx, cy = cam.params[0:4]
        fx = torch.tensor(fx, device=device, dtype=torch.float32)
        fy = torch.tensor(fy, device=device, dtype=torch.float32)
        cx = torch.tensor(cx, device=device, dtype=torch.float32)
        cy = torch.tensor(cy, device=device, dtype=torch.float32)

        # project all points
        X = pts.t()              # (3,N)
        X_cam = R @ X + t        # (3,N)

        Z = X_cam[2]             # (N,)
        valid_z = Z > 1e-6       # in front of camera

        Z_safe = torch.where(valid_z, Z, torch.ones_like(Z))

        u = fx * (X_cam[0] / Z_safe) + cx
        v = fy * (X_cam[1] / Z_safe) + cy

        u_int = u.long()
        v_int = v.long()

        valid_xy = (
            valid_z &
            (u_int >= 0) & (u_int < W) &
            (v_int >= 0) & (v_int < H)
        )

        if not valid_xy.any():
            continue

        idx_valid = torch.nonzero(valid_xy, as_tuple=False).squeeze(-1)  # (K,)

        v_sel = v_int[idx_valid]
        u_sel = u_int[idx_valid]
        mask_vals = mask[v_sel, u_sel]  # (K,)

        total[idx_valid] += 1
        outside[idx_valid] += (mask_vals == 0).to(torch.int32)

    total_f = total.to(torch.float32)
    outside_f = outside.to(torch.float32)

    ratio = torch.zeros_like(total_f)
    valid_points = total_f > 0
    ratio[valid_points] = outside_f[valid_points] / total_f[valid_points]

    ratio_np = ratio.cpu().numpy()
    return ratio_np


def compute_outside_ratio_pipeline(
    ply_path: str,
    sparse_dir: str,
    image_dir: str,
    ratio_out_path: str,
    device: str = "cuda",
    force_recompute: bool = False,
) -> np.ndarray:
    """
    High-level wrapper:
      - if ratio_out_path exists and not force_recompute: load it
      - else: compute outside_ratio and save to ratio_out_path
    """
    if (not force_recompute) and os.path.isfile(ratio_out_path):
        print(f"[RATIO] Using existing ratio file: {ratio_out_path}")
        ratio = np.load(ratio_out_path)
        print(f"[RATIO] Loaded ratio with shape {ratio.shape}")
        return ratio

    # read PLY
    ply = PlyData.read(ply_path)
    vertex = ply["vertex"]

    if not all(name in vertex.data.dtype.names for name in ("x", "y", "z")):
        raise RuntimeError(
            f"PLY does not contain x,y,z properties: {vertex.data.dtype.names}"
        )

    pts = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1)
    N = pts.shape[0]
    print(f"[RATIO] Loaded {N} points from PLY: {ply_path}")

    # load COLMAP
    cameras, images, _ = load_colmap_model(sparse_dir)
    images_list = list(images.values())
    print(f"[RATIO] Loaded {len(images_list)} COLMAP images from: {sparse_dir}")

    # load alpha masks
    mask_cache_np = {}
    for img in images_list:
        img_path = os.path.join(image_dir, img.name)
        rgba = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

        if rgba is None or rgba.shape[2] != 4:
            raise RuntimeError(
                f"Cannot read RGBA image (4 channels required): {img_path}"
            )

        alpha = rgba[:, :, 3]
        _, mask = cv2.threshold(alpha, 0, 255, cv2.THRESH_BINARY)
        mask_cache_np[img.id] = mask

    if device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA not available, falling back to CPU.")
        device = "cpu"

    ratio_np = compute_outside_ratio_torch(
        pts_np=pts,
        cameras=cameras,
        images=images,
        mask_cache_np=mask_cache_np,
        device=device,
    )

    os.makedirs(os.path.dirname(ratio_out_path), exist_ok=True)
    np.save(ratio_out_path, ratio_np)
    print(f"[RATIO] Saved outside_ratio to: {ratio_out_path}")
    print(f"[RATIO] min={ratio_np.min():.4f}, max={ratio_np.max():.4f}, mean={ratio_np.mean():.4f}")

    return ratio_np


# ======================
#  Step 2: 3DGS eval (local)
# ======================

@torch.no_grad()
def evaluate_ply_local(dataset, pipe, model_path, model_iteration, ply_path, max_views=None):
    """
    Evaluate the given ply_path (3DGS point cloud) on all train cameras:
    returns average L1 / PSNR / SSIM.
    """
    gaussians = GaussianModel(dataset.sh_degree, "adam")
    scene = Scene(dataset, gaussians)

    if not os.path.isfile(ply_path):
        raise FileNotFoundError(f"PLY not found: {ply_path}")

    print(f"[EVAL] Loading Gaussians from: {ply_path}")
    gaussians.load_ply(ply_path)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    train_cams = scene.getTrainCameras()
    if max_views is not None:
        train_cams = train_cams[:max_views]

    print(f"[EVAL] #Train views: {len(train_cams)}")

    total_l1 = 0.0
    total_psnr = 0.0
    total_ssim_val = 0.0
    num_views = 0

    for idx, viewpoint in enumerate(train_cams):
        render_pkg = render(
            viewpoint,
            gaussians,
            pipe,
            background,
            use_trained_exp=dataset.train_test_exp,
            separate_sh=False,
        )
        image = torch.clamp(render_pkg["render"], 0.0, 1.0)
        gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)

        if dataset.train_test_exp:
            image = image[..., image.shape[-1] // 2 :]
            gt_image = gt_image[..., gt_image.shape[-1] // 2 :]

        l1_val = l1_loss(image, gt_image).mean().double()
        psnr_val = psnr(image, gt_image).mean().double()

        if FUSED_SSIM_AVAILABLE:
            ssim_val = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0)).double()
        else:
            ssim_val = ssim(image, gt_image).double()

        total_l1 += l1_val.item()
        total_psnr += psnr_val.item()
        total_ssim_val += ssim_val.item()
        num_views += 1

    avg_l1 = total_l1 / max(1, num_views)
    avg_psnr = total_psnr / max(1, num_views)
    avg_ssim_val = total_ssim_val / max(1, num_views)

    print("\n========== 3DGS Train Set Evaluation ==========")
    print(f"#Train views:    {num_views}")
    print(f"Average L1:      {avg_l1:.6f}")
    print(f"Average PSNR:    {avg_psnr:.3f} dB")
    print(f"Average SSIM:    {avg_ssim_val:.4f}")
    print("===============================================")

    return {
        "avg_l1": avg_l1,
        "avg_psnr": avg_psnr,
        "avg_ssim": avg_ssim_val,
        "num_views": num_views,
    }


# ======================
#  Step 3: filter PLY using ratio
# ======================

def filter_ply_with_ratio(orig_ply_path, backup_ply_path, ratio, threshold):
    """
    Read original PLY from backup_ply_path, filter by ratio < threshold,
    and overwrite orig_ply_path.
    """
    ply = PlyData.read(backup_ply_path)
    vertex = ply["vertex"]
    N = len(vertex)

    if ratio.shape[0] != N:
        raise ValueError(f"ratio length {ratio.shape[0]} != #vertices {N}")

    keep = ratio < threshold
    kept_indices = np.where(keep)[0]
    new_vertices = vertex[kept_indices]

    new_ply = PlyData(
        [PlyElement.describe(new_vertices, "vertex")],
        text=ply.text,
    )
    new_ply.write(orig_ply_path)
    print(f"[SWEEP] threshold={threshold:.4f}, kept {len(new_vertices)}/{N} points")


# ======================
#  Step 4: Ratio / threshold 分析 + Knee detection
# ======================

def analyze_ratio_and_thresholds(ratio, results, out_dir: str):
    """
    Create:
      - ratio histogram
      - ratio CDF
      - threshold vs num_points curve + knee detection
    Save plots and write a small text summary.

    Args:
        ratio:   (N,) outside_ratio per point
        results: list of dicts:
                 { "threshold", "num_points", "avg_l1", "avg_psnr", "avg_ssim", "num_views" }
        out_dir: directory to save plots & summary
    """
    os.makedirs(out_dir, exist_ok=True)

    # 1) Ratio histogram
    plt.figure(figsize=(6, 4))
    plt.hist(ratio, bins=100, range=(0.0, 1.0), density=True)
    plt.xlabel("outside_ratio")
    plt.ylabel("density")
    plt.title("Histogram of outside_ratio")
    plt.tight_layout()
    hist_path = os.path.join(out_dir, "outside_ratio_histogram.png")
    plt.savefig(hist_path)
    plt.close()
    print(f"[ANALYSIS] Saved ratio histogram to: {hist_path}")

    # 2) Ratio CDF
    ratio_sorted = np.sort(ratio)
    cdf = np.linspace(0, 1, len(ratio_sorted), endpoint=True)
    plt.figure(figsize=(6, 4))
    plt.plot(ratio_sorted, cdf)
    plt.xlabel("outside_ratio (sorted)")
    plt.ylabel("CDF")
    plt.title("CDF of outside_ratio")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    cdf_path = os.path.join(out_dir, "outside_ratio_cdf.png")
    plt.savefig(cdf_path)
    plt.close()
    print(f"[ANALYSIS] Saved ratio CDF to: {cdf_path}")

    # 3) threshold vs num_points curve
    thresholds = np.array([r["threshold"] for r in results], dtype=np.float32)
    num_points = np.array([r["num_points"] for r in results], dtype=np.int64)

    # sort by threshold just in case
    sort_idx = np.argsort(thresholds)
    thresholds = thresholds[sort_idx]
    num_points = num_points[sort_idx]

    plt.figure(figsize=(6, 4))
    plt.plot(thresholds, num_points, marker="o")
    plt.xlabel("threshold")
    plt.ylabel("#points (kept)")
    plt.title("threshold vs #points")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    curve_path = os.path.join(out_dir, "threshold_vs_num_points.png")
    plt.savefig(curve_path)
    plt.close()
    print(f"[ANALYSIS] Saved threshold vs num_points curve to: {curve_path}")

    # 4) Knee detection on threshold-num_points curve（最大距离法）
    best_t = None
    if len(thresholds) >= 3:
        x = thresholds
        y = num_points.astype(np.float32)

        # line between (x0, y0) and (x_end, y_end)
        x0, y0 = x[0], y[0]
        x1, y1 = x[-1], y[-1]
        dx = x1 - x0
        dy = y1 - y0
        denom = np.sqrt(dx * dx + dy * dy) + 1e-8

        # distance from each point to that line
        distances = np.abs(dy * (x - x0) - dx * (y - y0)) / denom
        knee_idx = np.argmax(distances)
        best_t = float(x[knee_idx])

        # Plot with knee point highlighted
        plt.figure(figsize=(6, 4))
        plt.plot(x, y, marker="o", label="curve")
        plt.scatter([x[knee_idx]], [y[knee_idx]], c="r", s=80, label=f"knee ~ {best_t:.3f}")
        plt.xlabel("threshold")
        plt.ylabel("#points (kept)")
        plt.title("threshold vs #points (with knee)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        knee_plot_path = os.path.join(out_dir, "threshold_vs_num_points_knee.png")
        plt.savefig(knee_plot_path)
        plt.close()
        print(f"[ANALYSIS] Knee detection result: threshold ≈ {best_t:.4f}")
        print(f"[ANALYSIS] Saved knee plot to: {knee_plot_path}")
    else:
        print("[ANALYSIS] Not enough thresholds for knee detection (need >= 3).")

    # 5) Save a small textual summary
    summary_path = os.path.join(out_dir, "ratio_analysis_summary.txt")
    with open(summary_path, "w") as f:
        f.write("outside_ratio statistics:\n")
        f.write(f"min:  {ratio.min():.6f}\n")
        f.write(f"max:  {ratio.max():.6f}\n")
        f.write(f"mean: {ratio.mean():.6f}\n")
        f.write(f"median: {np.median(ratio):.6f}\n\n")

        f.write("threshold sweep (sorted):\n")
        f.write("threshold, num_points\n")
        for t, n in zip(thresholds, num_points):
            f.write(f"{t:.6f}, {int(n)}\n")

        if best_t is not None:
            f.write(f"\nEstimated knee threshold (max-distance method): {best_t:.6f}\n")
    print(f"[ANALYSIS] Saved summary to: {summary_path}")

    return best_t


# ======================
#  Main
# ======================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Full pipeline: compute outside_ratio for 3DGS PLY using RGBA alpha masks, "
            "then sweep thresholds and evaluate each filtered PLY on train views, "
            "optionally analyze ratio distribution and threshold curve."
        )
    )

    lp = ModelParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument(
        "--sparse_dir",
        type=str,
        required=True,
        help="COLMAP sparse model directory (cameras.bin, images.bin, points3D.bin).",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="Directory containing RGBA images (alpha used as mask).",
    )
    parser.add_argument(
        "--model_iteration",
        type=int,
        default=30000,
        help="Which iteration of point_cloud.ply to load (for path construction).",
    )
    parser.add_argument(
        "--ratio_path",
        type=str,
        default="",
        help=(
            "Path to save/load outside_ratio.npy. "
            "If empty, defaults to <model_path>/point_cloud/iteration_xxx/outside_ratio.npy"
        ),
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        required=True,
        help="List of outside_ratio thresholds to sweep, e.g. --thresholds 0.03 0.05 0.1",
    )
    parser.add_argument(
        "--max_views",
        type=int,
        default=None,
        help="If set, only evaluate first N train views (for speed).",
    )
    parser.add_argument(
        "--csv_out",
        type=str,
        default=None,
        help="Optional path to save CSV with metrics.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for computing outside_ratio.",
    )
    parser.add_argument(
        "--force_recompute_ratio",
        action="store_true",
        help="If set, recompute outside_ratio even if ratio_path already exists.",
    )
    parser.add_argument(
        "--export_filtered",
        action="store_true",
        help="Save each threshold's filtered PLY as separate file.",
    )
    parser.add_argument(
        "--analyze_ratio",
        action="store_true",
        help="If set, analyze ratio distribution and threshold curve, plot hist/CDF and knee.",
    )
    parser.add_argument(
        "--analysis_out_dir",
        type=str,
        default="",
        help="Directory to save analysis plots; default to ply iteration dir.",
    )

    args = parser.parse_args(sys.argv[1:])

    # same logic as 3DGS training script
    args.model_path = os.path.join(args.source_path, "output") if args.model_path == "" else args.model_path
    model_path = args.model_path
    print("Model path:", model_path)

    safe_state(False)

    dataset = lp.extract(args)
    pipe = pp.extract(args)

    # 3DGS point_cloud path
    ply_dir = os.path.join(
        model_path,
        "point_cloud",
        f"iteration_{args.model_iteration}",
    )
    orig_ply_path = os.path.join(ply_dir, "point_cloud.ply")

    if not os.path.isfile(orig_ply_path):
        raise FileNotFoundError(f"Original PLY not found: {orig_ply_path}")

    # ratio file path
    if args.ratio_path == "":
        ratio_path = os.path.join(ply_dir, "outside_ratio.npy")
    else:
        ratio_path = args.ratio_path

    # compute / load ratio
    ratio = compute_outside_ratio_pipeline(
        ply_path=orig_ply_path,
        sparse_dir=args.sparse_dir,
        image_dir=args.image_dir,
        ratio_out_path=ratio_path,
        device=args.device,
        force_recompute=args.force_recompute_ratio,
    )

    # backup original PLY
    backup_ply_path = orig_ply_path + ".backup"
    if not os.path.isfile(backup_ply_path):
        print(f"[SWEEP] Backing up original PLY to: {backup_ply_path}")
        shutil.copy2(orig_ply_path, backup_ply_path)
    else:
        print(f"[SWEEP] Using existing backup: {backup_ply_path}")

    # threshold sweep
    results = []

    for t in args.thresholds:
        print(f"\n====== Threshold {t:.4f} ======")
        # 1) write filtered PLY to orig_ply_path
        filter_ply_with_ratio(
            orig_ply_path=orig_ply_path,
            backup_ply_path=backup_ply_path,
            ratio=ratio,
            threshold=t,
        )
        # 2) eval
        metrics = evaluate_ply_local(
            dataset=dataset,
            pipe=pipe,
            model_path=model_path,
            model_iteration=args.model_iteration,
            ply_path=orig_ply_path,
            max_views=args.max_views,
        )
        # 3) record
        results.append({
            "threshold": t,
            "num_points": int((ratio < t).sum()),
            **metrics,
        })

        # optionally export this filtered PLY as a separate file
        if args.export_filtered:
            out_file = os.path.join(
                ply_dir, f"point_cloud_clean_t{t:.3f}.ply"
            )
            print(f"[SWEEP] Exporting filtered PLY for t={t:.3f} -> {out_file}")
            shutil.copy2(orig_ply_path, out_file)

    # restore original PLY
    print("\n[SWEEP] Restoring original PLY...")
    shutil.move(backup_ply_path, orig_ply_path)

    # print summary
    print("\n===== Sweep Summary =====")
    print("thresh\tpoints\tL1\t\tPSNR\tSSIM")
    for r in results:
        print(f"{r['threshold']:.4f}\t{r['num_points']}\t{r['avg_l1']:.6f}\t{r['avg_psnr']:.3f}\t{r['avg_ssim']:.4f}")

    # optional CSV
    if args.csv_out is not None:
        import csv
        os.makedirs(os.path.dirname(args.csv_out), exist_ok=True)
        with open(args.csv_out, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["threshold", "num_points", "avg_l1", "avg_psnr", "avg_ssim", "num_views"])
            for r in results:
                writer.writerow([
                    r["threshold"],
                    r["num_points"],
                    r["avg_l1"],
                    r["avg_psnr"],
                    r["avg_ssim"],
                    r["num_views"],
                ])
        print(f"[SWEEP] Saved CSV to: {args.csv_out}")

    # ratio / threshold analysis + knee detection
    if args.analyze_ratio:
        if args.analysis_out_dir == "":
            analysis_dir = os.path.join(ply_dir, "ratio_analysis")
        else:
            analysis_dir = args.analysis_out_dir

        print(f"[ANALYSIS] Running ratio / threshold analysis, saving to: {analysis_dir}")
        best_t = analyze_ratio_and_thresholds(
            ratio=ratio,
            results=results,
            out_dir=analysis_dir,
        )
        if best_t is not None:
            print(f"[ANALYSIS] Suggested knee threshold ≈ {best_t:.4f}")
        else:
            print("[ANALYSIS] Knee threshold could not be estimated (need >= 3 thresholds).")


if __name__ == "__main__":
    main()
