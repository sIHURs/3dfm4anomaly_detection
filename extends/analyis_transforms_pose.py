#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, json, numpy as np, matplotlib.pyplot as plt
import pandas as pd
import colmap

import torch

def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
        """
        Returns torch.sqrt(torch.max(0, x))
        but with a zero subgradient where x is 0.
        """
        ret = torch.zeros_like(x)
        positive_mask = x > 0
        ret[positive_mask] = torch.sqrt(x[positive_mask])
        return ret
    def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
        """
        Convert a unit quaternion to a standard form: one in which the real
        part is non negative.

        Args:
            quaternions: Quaternions with real part first,
                as tensor of shape (..., 4).

        Returns:
            Standardized quaternions as tensor of shape (..., 4).
        """
        return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)
    out = quat_candidates[
        torch.nn.functional.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))
    return standardize_quaternion(out)

def load_tf(path):
    d = json.load(open(path, "r"))
    frames = d["frames"]
    T = {f["file_path"]: np.array(f["transform_matrix"], float) for f in frames}
    return T

def umeyama_align(Y, X):
    # align Y->X，return s,R,t
    muX, muY = X.mean(0), Y.mean(0)
    X0, Y0 = X - muX, Y - muY
    U, S, Vt = np.linalg.svd(Y0.T @ X0 / Y.shape[0])
    R = U @ np.diag([1,1,np.sign(np.linalg.det(U @ Vt))]) @ Vt
    varY = (Y0**2).sum() / Y.shape[0]
    s = np.trace(np.diag(S) @ np.diag([1,1,np.sign(np.linalg.det(U @ Vt))])) / varY
    t = muX - s * (R @ muY)
    return s, R, t

def rot_err_deg(R1, R2):
    R = R1.T @ R2
    tr = np.clip((np.trace(R) - 1)/2, -1, 1)
    return np.degrees(np.arccos(tr))

def pack_T_from_RC(R, C, mode='c2w'):
    """
    R: (N,3,3), C: (N,3)
    return: T: (N,4,4)
    """
    N = R.shape[0]
    T = np.zeros((N, 4, 4), dtype=R.dtype)
    T[:, :3, :3] = R
    if mode == 'c2w':
        T[:, :3, 3] = C
    elif mode == 'w2c':
        # t = -R @ C
        T[:, :3, 3] = (-np.einsum('nij,nj->ni', R, C))
    else:
        raise ValueError("mode must be 'c2w' or 'w2c'")
    T[:, 3, 3] = 1.0
    return T

def draw_camera_axes(ax, C, R, convention='opencv', scale=None, draw_axes=False):
    C = np.asarray(C); R = np.asarray(R)
    if scale is None:
        if len(C) >= 2:
            diag = np.linalg.norm(C.max(0) - C.min(0))
            scale = max(diag * 0.05, 1e-6)
        else:
            scale = 0.1

    x_axis = R[:, :, 0]
    y_axis = R[:, :, 1]
    z_axis = R[:, :, 2]

    fwd = z_axis.copy()
    if convention.lower() == 'opengl':
        fwd = -fwd

    ax.quiver(C[:,0], C[:,1], C[:,2],
              fwd[:,0], fwd[:,1], fwd[:,2],
              length=scale, normalize=True,
              color='grey', linewidth=0.5)

    if draw_axes:
        ax.quiver(C[:,0], C[:,1], C[:,2],
                  x_axis[:,0], x_axis[:,1], x_axis[:,2],
                  length=scale*0.8, normalize=True)
        ax.quiver(C[:,0], C[:,1], C[:,2],
                  y_axis[:,0], y_axis[:,1], y_axis[:,2],
                  length=scale*0.8, normalize=True)
        ax.quiver(C[:,0], C[:,1], C[:,2],
                  fwd[:,0], fwd[:,1], fwd[:,2],
                  length=scale*0.8, normalize=True)
        
def save_transforms_like(in_json_path, out_json_path, T_map):
    with open(in_json_path, "r") as f:
        d = json.load(f)

    # 覆盖 frames 中的 transform_matrix
    replaced = 0
    for fr in d.get("frames", []):
        fp = fr.get("file_path")
        if fp in T_map:
            fr["transform_matrix"] = T_map[fp].tolist()
            replaced += 1

    if replaced == 0:
        print("No overlapping file_path between the two transforms.json")

    with open(out_json_path, "w") as f:
        json.dump(d, f, indent=2)
    print(f"[Save] {replaced} poses to: {out_json_path}")

def load_sim3_from_json(json_path):
    with open(json_path, "r") as f:
        d = json.load(f)

    s  = d.get("scale", d.get("s"))
    Rg = d.get("rotation", d.get("R", d.get("Rg")))
    tg = d.get("translation", d.get("t", d.get("tg")))

    if s is None or Rg is None or tg is None:
        raise ValueError(f"Invalid SIM(3) json: {json_path}. Required keys: scale/rotation/translation")

    s  = float(s)
    Rg = np.asarray(Rg, dtype=float)
    tg = np.asarray(tg, dtype=float).reshape(3)

    # 基本校验
    if Rg.shape != (3,3):
        raise ValueError(f"rotation must be 3x3, got {Rg.shape}")
    if not np.isfinite([s, *Rg.flatten(), *tg]).all():
        raise ValueError("SIM(3) contains non-finite values")
    return s, Rg, tg


def run_pose_analysis(dir, cls, align=True, save_aligned_pred=None, vis=False, save_sim3=False, use_sim3=False):

    pred = f"{dir}/{cls}/transforms_converted_openGL.json"
    gt   = f"{dir}/{cls}/transforms_train.json"

    T_pred = load_tf(pred)
    T_gt   = load_tf(gt)

    def normalize_key(k):
        import os
        base = os.path.basename(k)
        base = os.path.splitext(base)[0]
        return base

    pred_map = {normalize_key(k): k for k in T_pred.keys()}
    gt_map   = {normalize_key(k): k for k in T_gt.keys()}

    keys = sorted(set(pred_map.keys()) & set(gt_map.keys()))
    if not keys:
        raise SystemExit("No overlapping file_path between the two transforms.json")
    
    orig_pred_keys = [pred_map[k] for k in keys]
    orig_gt_keys   = [gt_map[k]   for k in keys]

    C_pred = np.stack([T_pred[k][:3,3] for k in orig_pred_keys], 0)
    R_pred = np.stack([T_pred[k][:3,:3] for k in orig_pred_keys], 0)
    C_gt   = np.stack([T_gt[k][:3,3]   for k in orig_gt_keys], 0)
    R_gt   = np.stack([T_gt[k][:3,:3]  for k in orig_gt_keys], 0)

    if align:
        if use_sim3:
            sim3_json = f"{dir}/{cls}/umeyama_align.json"
            s, Rg, tg = load_sim3_from_json(sim3_json)
            print(f"[Load] Using SIM(3) from {sim3_json} for alignment: s={s:.6f}, Rg=\n{Rg}, t={tg}")
        else:
            s, Rg, tg = umeyama_align(C_pred, C_gt)

            if save_sim3:
                align_dict = {
                    "scale": float(s),
                    "rotation": Rg.tolist(),
                    "translation": tg.tolist()
                }

                save_path = f"{dir}/{cls}/umeyama_align.json"
                with open(save_path, "w") as f:
                    json.dump(align_dict, f, indent=4)
                print(f"[Save] Umeyama alignment saved to {save_path}")

        C_pred = (s * (C_pred @ Rg.T)) + tg
        R_pred = np.einsum('ij,njk->nik', Rg, R_pred)
        print(f"[Align] scale={s:.6f}\n[Align] Rg=\n{Rg}\n[Align] t={tg}")


    if save_aligned_pred:
        T_pred_batch = pack_T_from_RC(R_pred, C_pred, mode='c2w')
        T_map = {k: T_pred_batch[i] for i, k in enumerate(keys)}

        save_transforms_like(pred, save_aligned_pred, T_map)

    # Splatpose Pose Estimation evaluation
    # TODO: implement splatpose error calculation like in the original paper

    # T_pred_batch = pack_T_from_RC(R_pred, C_pred, mode='c2w')
    # T_gt_batch   = pack_T_from_RC(R_gt,   C_gt,   mode='c2w')

    # T_pred_4x4 = {k: T_pred_batch[i] for i, k in enumerate(keys)}
    # T_gt_4x4   = {k: T_gt_batch[i]   for i, k in enumerate(keys)}

    # for key in keys:
    #     gt_quat = matrix_to_quaternion(torch.tensor(T_gt_4x4[key][:3, :3])).numpy()
    #     pred_quat = matrix_to_quaternion(torch.tensor(T_pred_4x4[key][:3, :3])).numpy()

    #     gt_trans = 
    #     trans_error = torch.sqrt(torch.sum(()))

    # TODO: or just do in 4x4 matrix form directly
    C_pred_t = torch.from_numpy(C_pred).float()  # (N,3)
    C_gt_t   = torch.from_numpy(C_gt).float()    # (N,3)
    t_err = torch.linalg.norm(C_gt_t - C_pred_t, dim=1) 

    R_pred_t = torch.from_numpy(R_pred).float()  # (N,3,3)
    R_gt_t   = torch.from_numpy(R_gt).float()    # (N,3,3)

    R_rel = torch.matmul(R_gt_t, R_pred_t.transpose(1,2))
    cosang = (torch.diagonal(R_rel, dim1=-2, dim2=-1).sum(-1) - 1) / 2
    cosang = torch.clamp(cosang, -1.0 + 1e-7, 1.0 - 1e-7)
    r_err  = torch.arccos(cosang)

    r_err_deg = r_err * 180.0 / torch.pi

    print(f"Splatpose Trans Error: mean={t_err.mean().item():.6g}, median={t_err.median().item():.6g}, max={t_err.max().item():.6g}")
    print(f"Splatpose Rot   Error: mean={r_err.mean().item():.6g}, median={r_err.median().item():.6g}, max={r_err.max().item():.6g}")
    print(f"Splatpose Rot   Error in deg: mean={r_err_deg.mean().item():.6g} deg, median={r_err_deg.median().item():.6g} deg, max={r_err_deg.max().item():.6g} deg")

    results = {
    "trans_error": {
        "mean": round(t_err.mean().item(), 3),
        "median": round(t_err.median().item(), 3),
        "max": round(t_err.max().item(), 3)
    },
    "rot_error_rad": {
        "mean": round(r_err.mean().item(), 3),
        "median": round(r_err.median().item(), 3),
        "max": round(r_err.max().item(), 3)
    },
    "rot_error_deg": {
        "mean": round(r_err_deg.mean().item(), 3),
        "median": round(r_err_deg.median().item(), 3),
        "max": round(r_err_deg.max().item(), 3)
    }
}

    if vis:
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(C_gt[:,0],   C_gt[:,1],   C_gt[:,2],   s=18, label="GT",   marker='o')
        ax.scatter(C_pred[:,0], C_pred[:,1], C_pred[:,2],
           s=14, label="Pred", marker='^', c='r')

        # error lines
        for a,b in zip(C_gt, C_pred):
            ax.plot([a[0],b[0]], [a[1],b[1]], [a[2],b[2]], linewidth=0.5)

        draw_camera_axes(ax, C_pred, R_pred, convention='opengl', draw_axes=False)

        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        ax.set_title("Camera centers: GT (o) vs Pred (^) " + ("[aligned]" if align else ""))
        ax.legend(); plt.tight_layout(); plt.show()

    return results


def main():
    ap = argparse.ArgumentParser("Compare two transforms.json (pred vs gt)")
    ap.add_argument("--dir", type=str, default="/home/yifan/studium/master_thesis/VGGT-X/vggtx_reconstruction/MAD-Sim_full")
    ap.add_argument("--align", action="store_true", help="Umeyama-align pred to GT (centers)")
    ap.add_argument("--vis", action="store_true", help="Visualize camera centers")
    ap.add_argument("--save_aligned_pred", action="store_true", help="If set, save the aligned pred transforms.json to this path")
    ap.add_argument("--class_name", type=str, default="01Gorilla", help="Class name for display only")
    ap.add_argument("--eval_all", action="store_true", help="If set, evaluate all sequences under the parent folder")
    ap.add_argument("--save_Eval", action="store_true", help="If set, save evaluation results to a csv file")
    ap.add_argument("--other_test", action="store_true", help="If set, use other test set")
    ap.add_argument("--save_sim3", action="store_true", help="If set, save sim3 aligned pred transforms.json")
    ap.add_argument("--use_sim3", action="store_true", help="If set, use sim3 aligned pred transforms.json for evaluation")
    args = ap.parse_args()

    classnames = ["01Gorilla", "02Unicorn", "03Mallard", "04Turtle", "05Whale", "06Bird", "07Owl", "08Sabertooth",
              "09Swan", "10Sheep", "11Pig", "12Zalika", "13Pheonix", "14Elephant", "15Parrot", "16Cat", "17Scorpion",
              "18Obesobeso", "19Bear", "20Puppy"]
    
    results_all = {}

    if args.other_test:
        result = run_pose_analysis(
            dir=args.dir,
            cls=args.class_name,
            align=args.align,
            save_aligned_pred=f"{args.dir}/{args.class_name}/pred_transforms_aligned.json" if args.save_aligned_pred else None,
            vis=args.vis,
            save_sim3=args.save_sim3,
            use_sim3=args.use_sim3
        )
        results_all[args.class_name] = result

    else:
    
        if args.eval_all:
            classes = classnames
        else:
            classes = [args.class_name]

        for cls in classes:
            print(f"\n[Evaluate] Class: {cls}\n[Evaluate] Pred: {args.dir}/{cls}/transforms_converted_opegGL.json \n")
            result = run_pose_analysis(
                dir=args.dir,
                cls=cls,
                align=args.align,
                save_aligned_pred=f"{args.dir}/{cls}/pred_transforms_aligned.json" if args.save_aligned_pred else None,
                vis=args.vis,
                save_sim3=args.save_sim3,
                use_sim3=args.use_sim3
            )
            results_all[cls] = result
        
    if args.save_Eval:
        # output as pandas DataFrame
        records = []
        for cls, res in results_all.items():
            flat = {"class": cls}
            for key, subdict in res.items():
                for metric, val in subdict.items():
                    flat[f"{key}_{metric}"] = val
            records.append(flat)

        df = pd.DataFrame(records)
        
        df.to_csv(f"{args.dir}/pose_eval_summary.csv", index=False)

        # save as latex
        latex = df.to_latex(index=False, escape=True, float_format="%.3f")

        latex_path = f"{args.dir}/table_pose_eval.tex"
        with open(latex_path, "w", encoding="utf-8") as f:
            f.write(latex)

if __name__ == "__main__":
    main()
