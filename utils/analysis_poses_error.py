#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os, re
import numpy as np
import torch
from typing import Optional

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.colors import Normalize
import matplotlib.cm as cm

DEBUG = True

# ----------------------------
# IO utils
# ----------------------------

def normalize_key(file_path: str) -> str:
    stem = os.path.splitext(os.path.basename(file_path))[0]
    m = re.search(r"(\d+)$", stem) or re.search(r"(\d+)", stem)
    if m:
        return str(int(m.group(1)))   # int(): "001" -> 1 -> "1"
    return stem

def build_key_map(T: dict[str, np.ndarray], name: str) -> dict[str, str]:
    """
    Build {normalized_key: original_file_path} and detect collisions.
    """
    m = {}
    for fp in T.keys():
        k = normalize_key(fp)
        if k in m:
            raise ValueError(
                f"[{name}] duplicate key after normalize: {k}\n"
                f"  A: {m[k]}\n"
                f"  B: {fp}\n"
                "Hint: your normalize_key() collapses two different frames to the same id."
            )
        m[k] = fp
    return m

def load_transforms_json(path: str) -> dict[str, np.ndarray]:
    """Load transforms.json into {file_path: (4,4) np.array}."""
    with open(path, "r") as f:
        d = json.load(f)
    frames = d.get("frames", [])
    if not frames:
        raise ValueError(f"No frames found in: {path}")
    T = {}
    for fr in frames:
        fp = fr.get("file_path")
        mat = fr.get("transform_matrix")
        if fp is None or mat is None:
            continue
        T[fp] = np.asarray(mat, dtype=float)
    if not T:
        raise ValueError(f"No valid (file_path, transform_matrix) pairs in: {path}")
    return T

def save_transforms_like(in_json_path: str, out_json_path: str, T_map: dict[str, np.ndarray]) -> None:
    """
    Copy the structure of in_json_path and overwrite transform_matrix where file_path matches keys in T_map.
    """
    with open(in_json_path, "r") as f:
        d = json.load(f)

    replaced = 0
    for fr in d.get("frames", []):
        fp = fr.get("file_path")
        if fp in T_map:
            fr["transform_matrix"] = T_map[fp].tolist()
            replaced += 1

    with open(out_json_path, "w") as f:
        json.dump(d, f, indent=2)

    print(f"[Save] replaced {replaced} poses -> {out_json_path}")


# ----------------------------
# Pose convention helpers
# ----------------------------
def extract_RC_from_T(T: np.ndarray, mode: str):
    """
    Extract camera center C (world coords) and rotation R_c2w from 4x4 matrices.

    mode = 'c2w':
        x_w = R_c2w x_c + C
        T = [R_c2w | C]
    mode = 'w2c':
        x_c = R_wc x_w + t_wc
        camera center C = -R_wc^T t_wc
        R_c2w = R_wc^T
    Return:
        R_c2w: (N,3,3)
        C:     (N,3)
    """
    T = np.asarray(T, dtype=float)
    if T.ndim == 2:
        T = T[None, ...]
    if T.shape[1:] != (4, 4):
        raise ValueError(f"T must be (N,4,4) or (4,4), got {T.shape}")

    R = T[:, :3, :3]
    t = T[:, :3, 3]

    mode = mode.lower()
    if mode == "c2w":
        R_c2w = R
        C = t
    elif mode == "w2c":
        R_c2w = np.transpose(R, (0, 2, 1))  # R_wc^T
        C = -np.einsum("nij,nj->ni", R_c2w, t)  # -R_wc^T t_wc
    else:
        raise ValueError("mode must be 'c2w' or 'w2c'")

    return R_c2w, C

def pack_T_c2w(R_c2w: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Pack as c2w: T = [R_c2w | C]."""
    R_c2w = np.asarray(R_c2w)
    C = np.asarray(C)
    N = R_c2w.shape[0]
    T = np.zeros((N, 4, 4), dtype=R_c2w.dtype)
    T[:, :3, :3] = R_c2w
    T[:, :3, 3] = C
    T[:, 3, 3] = 1.0
    return T

def pack_T_w2c_from_c2w(R_c2w: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Convert (R_c2w, C) to w2c matrix: R_wc=R_c2w^T, t_wc=-R_wc*C."""
    R_c2w = np.asarray(R_c2w)
    C = np.asarray(C)
    R_wc = np.transpose(R_c2w, (0, 2, 1))
    t_wc = -np.einsum("nij,nj->ni", R_wc, C)

    N = R_wc.shape[0]
    T = np.zeros((N, 4, 4), dtype=R_wc.dtype)
    T[:, :3, :3] = R_wc
    T[:, :3, 3] = t_wc
    T[:, 3, 3] = 1.0
    return T


# ----------------------------
# Geometry: Umeyama SIM(3)
# ----------------------------
def umeyama_align(Y: np.ndarray, X: np.ndarray):
    """
    Align Y -> X via SIM(3). Return s, R, t.
    Points are row-vectors here, and we apply: Y_aligned = s * (Y @ R.T) + t
    """
    if Y.shape != X.shape or Y.ndim != 2 or Y.shape[1] != 3:
        raise ValueError(f"Y and X must be (N,3) with same shape, got Y={Y.shape}, X={X.shape}")

    muX, muY = X.mean(0), Y.mean(0)
    X0, Y0 = X - muX, Y - muY

    U, S, Vt = np.linalg.svd(Y0.T @ X0 / Y.shape[0])
    det = np.linalg.det(U @ Vt)
    D = np.diag([1.0, 1.0, np.sign(det)])
    R = U @ D @ Vt

    varY = (Y0 ** 2).sum() / Y.shape[0]
    if varY <= 1e-12:
        raise ValueError("Degenerate alignment: variance of Y is too small (varY ~ 0).")

    s = np.trace(np.diag(S) @ D) / varY
    t = muX - s * (R @ muY)

    return float(s), R, t

def save_sim3(sim3_path: str, s: float, R: np.ndarray, t: np.ndarray) -> None:
    d = {"scale": float(s), "rotation": R.tolist(), "translation": t.reshape(3).tolist()}
    with open(sim3_path, "w") as f:
        json.dump(d, f, indent=4)
    print(f"[Save] SIM(3) -> {sim3_path}")

def load_sim3(sim3_path: str):
    with open(sim3_path, "r") as f:
        d = json.load(f)

    s = d.get("scale", d.get("s"))
    R = d.get("rotation", d.get("R", d.get("Rg")))
    t = d.get("translation", d.get("t", d.get("tg")))

    if s is None or R is None or t is None:
        raise ValueError(f"Invalid SIM(3) json: {sim3_path}. Need keys scale/rotation/translation.")

    s = float(s)
    R = np.asarray(R, dtype=float)
    t = np.asarray(t, dtype=float).reshape(3)

    if R.shape != (3, 3):
        raise ValueError(f"rotation must be 3x3, got {R.shape}")
    if not np.isfinite([s, *R.flatten(), *t]).all():
        raise ValueError("SIM(3) contains non-finite values")

    return s, R, t


# ----------------------------
# Visualization helpers
# ----------------------------
def draw_camera_axes(ax, C, R_c2w, convention="opengl", scale=None):
    """Draw forward direction (z axis) as quiver (assumes R is c2w)."""
    C = np.asarray(C)
    R = np.asarray(R_c2w)
    if scale is None:
        diag = np.linalg.norm(C.max(0) - C.min(0)) if len(C) >= 2 else 1.0
        scale = max(diag * 0.05, 1e-6)

    # In c2w, camera axes in world are columns of R. z-axis = R[:, :, 2]
    z_axis = R[:, :, 2]
    fwd = -z_axis if convention.lower() == "opengl" else z_axis

    ax.quiver(
        C[:, 0], C[:, 1], C[:, 2],
        fwd[:, 0], fwd[:, 1], fwd[:, 2],
        length=scale, normalize=True,
        color="grey", linewidth=0.5
    )


# ----------------------------
# Robust alignment helpers
# ----------------------------
def _apply_sim3_to_centers(C: np.ndarray, s: float, Rg: np.ndarray, tg: np.ndarray) -> np.ndarray:
    return (s * (C @ Rg.T)) + tg

def robust_umeyama_align(
    C_pr: np.ndarray,
    C_gt: np.ndarray,
    mode: str = "trim",          # "trim" or "mad"
    trim_q: float = 0.90,        # keep best q fraction
    mad_k: float = 2.5,          # keep <= median + k*MAD
    min_inliers: int = 10,
):
    """
    Two-stage robust SIM(3) alignment using camera centers.
    1) Fit SIM(3) on all pairs
    2) Compute residuals after alignment
    3) Select inliers (trim or MAD)
    4) Re-fit SIM(3) on inliers
    Return: (s, Rg, tg, inlier_mask, residuals_stage1)
    """
    N = C_pr.shape[0]
    if N < 3:
        raise ValueError("Need at least 3 matched frames for robust alignment.")

    # stage 1
    s1, R1, t1 = umeyama_align(C_pr, C_gt)
    C1 = _apply_sim3_to_centers(C_pr, s1, R1, t1)
    resid = np.linalg.norm(C1 - C_gt, axis=1)

    # inlier selection
    mode = mode.lower()
    if mode == "trim":
        q = float(trim_q)
        q = max(0.0, min(1.0, q))
        if q <= 0:
            raise ValueError("trim_q must be > 0.")
        thr = float(np.quantile(resid, q))
        inliers = resid <= thr
    elif mode == "mad":
        med = float(np.median(resid))
        mad = float(np.median(np.abs(resid - med)))
        mad = max(mad, 1e-12)
        thr = med + float(mad_k) * mad
        inliers = resid <= thr
    else:
        raise ValueError("robust_mode must be 'trim' or 'mad'.")

    num_in = int(inliers.sum())
    if num_in < min_inliers:
        idx = np.argsort(resid)[:min_inliers]
        inliers = np.zeros_like(inliers, dtype=bool)
        inliers[idx] = True

    # stage 2 on inliers
    s2, R2, t2 = umeyama_align(C_pr[inliers], C_gt[inliers])

    return s2, R2, t2, inliers, resid


# ----------------------------
# Metrics helpers
# ----------------------------
def _stats_from_tensor(x: torch.Tensor) -> dict:
    x = x.detach().cpu()
    # robust percentiles
    q75 = torch.quantile(x, 0.75).item()
    q90 = torch.quantile(x, 0.90).item()
    q95 = torch.quantile(x, 0.95).item()
    q99 = torch.quantile(x, 0.99).item()
    return {
        "mean": x.mean().item(),
        "median": x.median().item(),
        "min": x.min().item(),
        "max": x.max().item(),
        "p75": q75,
        "p90": q90,
        "p95": q95,
        "p99": q99,
    }

def _rmse_from_l2_errors(err: torch.Tensor) -> float:
    # err is L2 per-sample
    return torch.sqrt(torch.mean(err ** 2)).item()


# ----------------------------
# Core evaluation
# ----------------------------
def evaluate_poses(
    gt_json: str,
    pred_json: str,

    gt_mode: str = "c2w",
    pred_mode: str = "c2w",
    save_mode: str = "c2w",

    align: bool = False,
    vis: bool = False,
    sim3_json: str = "./umeyama_align.json",
    save_sim3_flag: bool = False,
    use_sim3: bool = False,
    save_aligned_pred: Optional[str] = None,

    robust_align: bool = False,
    robust_mode: str = "trim",
    trim_q: float = 0.90,
    mad_k: float = 2.5,
    outlier_topk: int = 20,
    save_inliers_json: Optional[str] = None,
):
    T_gt_dict = load_transforms_json(gt_json)
    T_pr_dict = load_transforms_json(pred_json)

    gt_total = len(T_gt_dict)
    pred_total = len(T_pr_dict)

    # 1) 重建成功率：pred / gt
    recon_success = (pred_total / gt_total) if gt_total > 0 else 0.0

    gt_map = build_key_map(T_gt_dict, "gt")
    pr_map = build_key_map(T_pr_dict, "pred")

    if DEBUG:
        gt_keys = set(gt_map.keys())
        pr_keys = set(pr_map.keys())
        common = gt_keys & pr_keys
        only_gt = gt_keys - pr_keys
        only_pr = pr_keys - gt_keys

        print(f"[DEBUG] #gt_keys={len(gt_keys)}  #pred_keys={len(pr_keys)}  #common={len(common)}")
        print(f"[DEBUG] only_gt={len(only_gt)}  only_pred={len(only_pr)}")

        print("[DEBUG] common sample:", sorted(common)[:20])
        print("[DEBUG] only_gt sample:", sorted(only_gt)[:20])
        print("[DEBUG] only_pred sample:", sorted(only_pr)[:20])

    keys = sorted(set(gt_map.keys()) & set(pr_map.keys()))
    if not keys:
        raise SystemExit("No overlapping frames after basename matching (without extension).")

    gt_fps = [gt_map[k] for k in keys]
    pr_fps = [pr_map[k] for k in keys]

    # Stack 4x4
    Tgt = np.stack([T_gt_dict[fp] for fp in gt_fps], 0)
    Tpr = np.stack([T_pr_dict[fp] for fp in pr_fps], 0)

    # Convert BOTH to unified representation: (R_c2w, C)
    R_gt, C_gt = extract_RC_from_T(Tgt, mode=gt_mode)
    R_pr, C_pr = extract_RC_from_T(Tpr, mode=pred_mode)

    # -------- alignment (optional) --------
    if align:
        if use_sim3:
            s, Rg, tg = load_sim3(sim3_json)
            print(f"[Load] SIM(3) from {sim3_json}: s={s:.6f}, t={tg}")
            inliers = None
            resid1 = None
        else:
            if robust_align:
                s, Rg, tg, inliers, resid1 = robust_umeyama_align(
                    C_pr, C_gt,
                    mode=robust_mode,
                    trim_q=trim_q,
                    mad_k=mad_k,
                    min_inliers=max(10, int(0.2 * len(C_pr))) if len(C_pr) >= 50 else max(10, len(C_pr)//2),
                )
                num_in = int(inliers.sum())
                print(f"[Robust Umeyama] mode={robust_mode}  inliers={num_in}/{len(C_pr)}")
                print(f"[Robust Umeyama] s={s:.6f}\nRg=\n{Rg}\nt={tg}")

                if resid1 is not None and outlier_topk > 0:
                    k = min(int(outlier_topk), len(resid1))
                    idx_sorted = np.argsort(resid1)[::-1]
                    worst = idx_sorted[:k]
                    print(f"[Robust Umeyama] Top-{k} residuals after stage-1 align:")
                    for j in worst:
                        tag = "OUTLIER" if (inliers is not None and not inliers[j]) else "inlier?"
                        print(f"  {tag:7s}  resid={resid1[j]:.6g}  pred_fp={pr_fps[j]}  gt_fp={gt_fps[j]}")

                if save_inliers_json is not None and resid1 is not None and inliers is not None:
                    payload = {
                        "keys": keys,
                        "gt_file_paths": gt_fps,
                        "pred_file_paths": pr_fps,
                        "stage1_residual": resid1.tolist(),
                        "inlier_mask": inliers.astype(bool).tolist(),
                        "robust_mode": robust_mode,
                        "trim_q": float(trim_q),
                        "mad_k": float(mad_k),
                    }
                    os.makedirs(os.path.dirname(save_inliers_json) or ".", exist_ok=True)
                    with open(save_inliers_json, "w") as f:
                        json.dump(payload, f, indent=2)
                    print(f"[Save] inliers/residuals -> {save_inliers_json}")

            else:
                s, Rg, tg = umeyama_align(C_pr, C_gt)
                inliers = None
                resid1 = None
                print(f"[Umeyama] s={s:.6f}\nRg=\n{Rg}\nt={tg}")

            if save_sim3_flag:
                save_sim3(sim3_json, s, Rg, tg)

        # Apply SIM(3) to centers
        C_pr = (s * (C_pr @ Rg.T)) + tg
        # Apply global rotation to orientations (c2w): R_c2w <- Rg @ R_c2w
        R_pr = np.einsum("ij,njk->nik", Rg, R_pr)

    # -------- compute errors --------
    C_pr_t = torch.from_numpy(C_pr).float()
    C_gt_t = torch.from_numpy(C_gt).float()

    # per-frame translation error (L2)
    t_err = torch.linalg.norm(C_gt_t - C_pr_t, dim=1)

    # 2) ATE (常用 RMSE 指标)
    ate_rmse = _rmse_from_l2_errors(t_err)
    ate_stats = _stats_from_tensor(t_err)

    # Rotation error for c2w:
    # R_rel = R_gt * R_pr^T  (since both are c2w)
    R_pr_t = torch.from_numpy(R_pr).float()
    R_gt_t = torch.from_numpy(R_gt).float()
    R_rel = torch.matmul(R_gt_t, R_pr_t.transpose(1, 2))

    cosang = (torch.diagonal(R_rel, dim1=-2, dim2=-1).sum(-1) - 1) / 2
    cosang = torch.clamp(cosang, -1.0 + 1e-7, 1.0 - 1e-7)
    r_err = torch.arccos(cosang)
    r_err_deg = r_err * 180.0 / torch.pi

    rot_stats_deg = _stats_from_tensor(r_err_deg)
    rot_stats_rad = _stats_from_tensor(r_err)

    matched = len(keys)
    match_rate_gt = matched / gt_total if gt_total > 0 else 0.0
    match_rate_pred = matched / pred_total if pred_total > 0 else 0.0

    print(f"[Counts] gt_frames={gt_total}  pred_frames={pred_total}")
    print(f"[Reconstruction success] pred/gt = {recon_success:.4f} ({pred_total}/{gt_total})")
    print(f"[Frames matched] {matched} / gt={gt_total} pred={pred_total}")
    print(f"[Coverage] matched/gt = {match_rate_gt:.4f}   matched/pred = {match_rate_pred:.4f}")

    # ATE 输出（更标准）
    print(
        f"ATE (translation L2): RMSE={ate_rmse:.6g}, "
        f"mean={ate_stats['mean']:.6g}, median={ate_stats['median']:.6g}, "
        f"p90={ate_stats['p90']:.6g}, p95={ate_stats['p95']:.6g}, "
        f"min={ate_stats['min']:.6g}, max={ate_stats['max']:.6g}"
    )

    print(
        f"Rot Error (deg): mean={rot_stats_deg['mean']:.6g}, median={rot_stats_deg['median']:.6g}, "
        f"p90={rot_stats_deg['p90']:.6g}, p95={rot_stats_deg['p95']:.6g}, "
        f"min={rot_stats_deg['min']:.6g}, max={rot_stats_deg['max']:.6g}"
    )
    print(
        f"Rot Error (rad): mean={rot_stats_rad['mean']:.6g}, median={rot_stats_rad['median']:.6g}, "
        f"p90={rot_stats_rad['p90']:.6g}, p95={rot_stats_rad['p95']:.6g}, "
        f"min={rot_stats_rad['min']:.6g}, max={rot_stats_rad['max']:.6g}"
    )

    results = {
        "counts": {
            "gt_frames": int(gt_total),
            "pred_frames": int(pred_total),
            "matched_frames": int(matched),
        },
        "rates": {
            "reconstruction_success_pred_over_gt": float(recon_success),
            "coverage_matched_over_gt": float(match_rate_gt),
            "coverage_matched_over_pred": float(match_rate_pred),
        },
        "ATE_translation": {
            "rmse": round(float(ate_rmse), 6),
            **{k: round(float(v), 6) for k, v in ate_stats.items()},
        },
        "rot_error_deg": {k: round(float(v), 6) for k, v in rot_stats_deg.items()},
        "rot_error_rad": {k: round(float(v), 6) for k, v in rot_stats_rad.items()},
        "meta": {
            "gt_mode_in": gt_mode,
            "pred_mode_in": pred_mode,
            "save_mode_out": save_mode if save_aligned_pred is not None else None,
            "aligned": bool(align),
        }
    }

    # -------- optionally save aligned pred transforms --------
    if save_aligned_pred is not None:
        if save_mode.lower() == "c2w":
            T_pr_batch = pack_T_c2w(R_pr, C_pr)
        elif save_mode.lower() == "w2c":
            T_pr_batch = pack_T_w2c_from_c2w(R_pr, C_pr)
        else:
            raise ValueError("save_mode must be 'c2w' or 'w2c'")

        # overwrite with original pred file_path keys
        T_map = {pr_fps[i]: T_pr_batch[i] for i in range(len(keys))}
        save_transforms_like(pred_json, save_aligned_pred, T_map)

    # -------- visualization (optional) --------
    if vis:
        pos_err = np.linalg.norm(C_pr - C_gt, axis=1)
        rot_deg_np = r_err_deg.detach().cpu().numpy()

        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection="3d")

        ax.scatter(C_pr[:, 0], C_pr[:, 1], C_pr[:, 2],
                   s=10, c="gray", alpha=0.25, marker="^", label="Pred (faint)")

        segments, widths, colors = [], [], []
        norm = Normalize(vmin=float(rot_deg_np.min()), vmax=float(rot_deg_np.max()))
        cmap = cm.get_cmap("plasma")
        pos_err_max = float(pos_err.max()) if float(pos_err.max()) > 0 else 1.0

        for gt, pr, perr, rdeg in zip(C_gt, C_pr, pos_err, rot_deg_np):
            segments.append([gt, pr])
            widths.append(0.5 + 2.5 * (perr / pos_err_max))
            colors.append(cmap(norm(rdeg)))

        lc = Line3DCollection(segments, colors=colors, linewidths=widths, alpha=0.9)
        ax.add_collection3d(lc)

        draw_camera_axes(ax, C_pr, R_pr, convention="opengl")

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(
            f"Pose Evaluation\n"
            f"color = rotation error (deg), line width = translation error "
            + ("[aligned]" if align else "")
        )

        mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
        mappable.set_array([])
        cbar = plt.colorbar(mappable, ax=ax, fraction=0.03, pad=0.05)
        cbar.set_label("Rotation error (deg)")

        ax.legend(loc="upper right")
        plt.tight_layout()
        plt.show()

    return results


def main():
    ap = argparse.ArgumentParser("Pose evaluation: gt.json vs estimated.json")
    ap.add_argument("--gt_json", type=str, required=True, help="Path to GT transforms.json")
    ap.add_argument("--pred_json", type=str, required=True, help="Path to estimated/pred transforms.json")

    # pose convention controls
    ap.add_argument("--gt_mode", type=str, default="c2w", choices=["c2w", "w2c"],
                    help="Pose convention of GT transforms.json (default: c2w)")
    ap.add_argument("--pred_mode", type=str, default="c2w", choices=["c2w", "w2c"],
                    help="Pose convention of Pred transforms.json (default: c2w)")
    ap.add_argument("--save_mode", type=str, default="c2w", choices=["c2w", "w2c"],
                    help="Pose convention to write when --save_aligned_pred is set (default: c2w)")

    ap.add_argument("--align", action="store_true", help="Umeyama-align pred to GT (camera centers)")
    ap.add_argument("--vis", action="store_true", help="Visualize camera centers and errors")
    ap.add_argument("--save_sim3", action="store_true", help="Save SIM(3) alignment to --sim3_json")
    ap.add_argument("--use_sim3", action="store_true", help="Load SIM(3) from --sim3_json for alignment")

    ap.add_argument("--sim3_json", type=str, default="./umeyama_align.json",
                    help="Where to save/load SIM(3) json (default: ./umeyama_align.json)")
    ap.add_argument("--save_aligned_pred", type=str, default=None,
                    help="If set, save aligned pred transforms.json to this path")

    ap.add_argument("--robust_align", action="store_true",
                    help="Enable robust 2-stage alignment: coarse SIM(3) -> outlier rejection -> refine SIM(3)")
    ap.add_argument("--robust_mode", type=str, default="trim", choices=["trim", "mad"],
                    help="Outlier rejection mode: 'trim' keeps best q fraction; 'mad' uses median+k*MAD")
    ap.add_argument("--trim_q", type=float, default=0.90,
                    help="For robust_mode=trim: keep best q fraction (default 0.90)")
    ap.add_argument("--mad_k", type=float, default=2.5,
                    help="For robust_mode=mad: threshold = median + k*MAD (default 2.5)")
    ap.add_argument("--outlier_topk", type=int, default=20,
                    help="Print top-k worst residuals after stage-1 align (default 20)")
    ap.add_argument("--save_inliers_json", type=str, default=None,
                    help="If set, save stage-1 residuals and inlier mask to this json path")

    args = ap.parse_args()

    evaluate_poses(
        gt_json=args.gt_json,
        pred_json=args.pred_json,
        gt_mode=args.gt_mode,
        pred_mode=args.pred_mode,
        save_mode=args.save_mode,

        align=args.align,
        vis=args.vis,
        sim3_json=args.sim3_json,
        save_sim3_flag=args.save_sim3,
        use_sim3=args.use_sim3,
        save_aligned_pred=args.save_aligned_pred,

        robust_align=args.robust_align,
        robust_mode=args.robust_mode,
        trim_q=args.trim_q,
        mad_k=args.mad_k,
        outlier_topk=args.outlier_topk,
        save_inliers_json=args.save_inliers_json,
    )


if __name__ == "__main__":
    main()
