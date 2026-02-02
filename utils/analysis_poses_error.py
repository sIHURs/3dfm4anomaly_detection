#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os, re
import numpy as np
import torch

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
        return str(int(m.group(1)))   # int() ： "001"->1->"1"

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


# def normalize_key(file_path: str) -> str:
#     """Match by basename without extension, e.g. /a/b/000.png -> 000"""
#     base = os.path.basename(file_path)
#     base = os.path.splitext(base)[0]
#     return base


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


def pack_T_from_RC(R: np.ndarray, C: np.ndarray, mode="c2w") -> np.ndarray:
    """R:(N,3,3), C:(N,3) -> T:(N,4,4)"""
    N = R.shape[0]
    T = np.zeros((N, 4, 4), dtype=R.dtype)
    T[:, :3, :3] = R
    if mode == "c2w":
        T[:, :3, 3] = C
    elif mode == "w2c":
        T[:, :3, 3] = -np.einsum("nij,nj->ni", R, C)
    else:
        raise ValueError("mode must be 'c2w' or 'w2c'")
    T[:, 3, 3] = 1.0
    return T


def draw_camera_axes(ax, C, R, convention="opengl", scale=None):
    """Draw forward direction (z axis) as quiver."""
    C = np.asarray(C)
    R = np.asarray(R)
    if scale is None:
        diag = np.linalg.norm(C.max(0) - C.min(0)) if len(C) >= 2 else 1.0
        scale = max(diag * 0.05, 1e-6)

    z_axis = R[:, :, 2]
    fwd = -z_axis if convention.lower() == "opengl" else z_axis

    ax.quiver(
        C[:, 0], C[:, 1], C[:, 2],
        fwd[:, 0], fwd[:, 1], fwd[:, 2],
        length=scale, normalize=True,
        color="grey", linewidth=0.5
    )


# ----------------------------
# Core evaluation
# ----------------------------
def evaluate_poses(
    gt_json: str,
    pred_json: str,
    align: bool = False,
    vis: bool = False,
    sim3_json: str = "./umeyama_align.json",
    save_sim3_flag: bool = False,
    use_sim3: bool = False,
    save_aligned_pred: str | None = None,
):
    T_gt = load_transforms_json(gt_json)
    T_pr = load_transforms_json(pred_json)

    # gt_map = {normalize_key(k): k for k in T_gt.keys()}
    # pr_map = {normalize_key(k): k for k in T_pr.keys()}
    gt_map = build_key_map(T_gt, "gt")
    pr_map = build_key_map(T_pr, "pred")

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

        print("\n[DEBUG] gt key->file_path (sample):")
        for k in sorted(gt_map.keys())[:20]:
            print(f"  {k} -> {gt_map[k]}")

        print("\n[DEBUG] pred key->file_path (sample):")
        for k in sorted(pr_map.keys())[:20]:
            print(f"  {k} -> {pr_map[k]}")

    keys = sorted(set(gt_map.keys()) & set(pr_map.keys()))
    if not keys:
        raise SystemExit("No overlapping frames after basename matching (without extension).")

    gt_keys = [gt_map[k] for k in keys]
    pr_keys = [pr_map[k] for k in keys]

    C_gt = np.stack([T_gt[k][:3, 3] for k in gt_keys], 0)
    R_gt = np.stack([T_gt[k][:3, :3] for k in gt_keys], 0)

    C_pr = np.stack([T_pr[k][:3, 3] for k in pr_keys], 0)
    R_pr = np.stack([T_pr[k][:3, :3] for k in pr_keys], 0)

    # -------- alignment (optional) --------
    if align:
        if use_sim3:
            s, Rg, tg = load_sim3(sim3_json)
            print(f"[Load] SIM(3) from {sim3_json}: s={s:.6f}, t={tg}")
        else:
            s, Rg, tg = umeyama_align(C_pr, C_gt)
            print(f"[Umeyama] s={s:.6f}\nRg=\n{Rg}\nt={tg}")
            if save_sim3_flag:
                save_sim3(sim3_json, s, Rg, tg)

        # apply to centers (row-vector convention)
        C_pr = (s * (C_pr @ Rg.T)) + tg
        # apply global rotation to camera orientation
        R_pr = np.einsum("ij,njk->nik", Rg, R_pr)

    # -------- compute errors --------
    C_pr_t = torch.from_numpy(C_pr).float()
    C_gt_t = torch.from_numpy(C_gt).float()
    t_err = torch.linalg.norm(C_gt_t - C_pr_t, dim=1)  # (N,)

    R_pr_t = torch.from_numpy(R_pr).float()
    R_gt_t = torch.from_numpy(R_gt).float()
    R_rel = torch.matmul(R_gt_t, R_pr_t.transpose(1, 2))
    cosang = (torch.diagonal(R_rel, dim1=-2, dim2=-1).sum(-1) - 1) / 2
    cosang = torch.clamp(cosang, -1.0 + 1e-7, 1.0 - 1e-7)
    r_err = torch.arccos(cosang)                 # rad
    r_err_deg = r_err * 180.0 / torch.pi         # deg

    print(f"[Frames matched] {len(keys)} / gt={len(T_gt)} pred={len(T_pr)}")
    print(f"Trans Error: mean={t_err.mean().item():.6g}, median={t_err.median().item():.6g}, max={t_err.max().item():.6g}")
    print(f"Rot   Error: mean={r_err.mean().item():.6g} rad, median={r_err.median().item():.6g} rad, max={r_err.max().item():.6g} rad")
    print(f"Rot   Error: mean={r_err_deg.mean().item():.6g} deg, median={r_err_deg.median().item():.6g} deg, max={r_err_deg.max().item():.6g} deg")

    results = {
        "trans_error": {
            "mean": round(t_err.mean().item(), 3),
            "median": round(t_err.median().item(), 3),
            "max": round(t_err.max().item(), 3),
        },
        "rot_error_rad": {
            "mean": round(r_err.mean().item(), 3),
            "median": round(r_err.median().item(), 3),
            "max": round(r_err.max().item(), 3),
        },
        "rot_error_deg": {
            "mean": round(r_err_deg.mean().item(), 3),
            "median": round(r_err_deg.median().item(), 3),
            "max": round(r_err_deg.max().item(), 3),
        },
    }

    # -------- optionally save aligned pred transforms --------
    if save_aligned_pred is not None:
        T_pr_batch = pack_T_from_RC(R_pr, C_pr, mode="c2w")
        # IMPORTANT: must use original pred file_path keys to overwrite
        T_map = {pr_keys[i]: T_pr_batch[i] for i in range(len(keys))}
        save_transforms_like(pred_json, save_aligned_pred, T_map)

    # -------- visualization (optional) --------
    if vis:
        pos_err = np.linalg.norm(C_pr - C_gt, axis=1)
        rot_deg_np = r_err_deg.detach().cpu().numpy()

        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection="3d")

        # faint pred points
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

    # keep the 4 flags you requested
    ap.add_argument("--align", action="store_true", help="Umeyama-align pred to GT (camera centers)")
    ap.add_argument("--vis", action="store_true", help="Visualize camera centers and errors")
    ap.add_argument("--save_sim3", action="store_true", help="Save SIM(3) alignment to --sim3_json")
    ap.add_argument("--use_sim3", action="store_true", help="Load SIM(3) from --sim3_json for alignment")

    # extra but useful knobs (optional)
    ap.add_argument("--sim3_json", type=str, default="./umeyama_align.json",
                    help="Where to save/load SIM(3) json (default: ./umeyama_align.json)")
    ap.add_argument("--save_aligned_pred", type=str, default=None,
                    help="If set, save aligned pred transforms.json to this path")

    args = ap.parse_args()

    evaluate_poses(
        gt_json=args.gt_json,
        pred_json=args.pred_json,
        align=args.align,
        vis=args.vis,
        sim3_json=args.sim3_json,
        save_sim3_flag=args.save_sim3,
        use_sim3=args.use_sim3,
        save_aligned_pred=args.save_aligned_pred,
    )


if __name__ == "__main__":
    main()
