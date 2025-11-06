#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, json, numpy as np, matplotlib.pyplot as plt

def load_tf(path):
    d = json.load(open(path, "r"))
    frames = d["frames"]
    T = {f["file_path"]: np.array(f["transform_matrix"], float) for f in frames}
    return T

def umeyama_align(Y, X):
    # 对齐 Y->X，返回 s,R,t
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

def main():
    ap = argparse.ArgumentParser("Compare two transforms.json (pred vs gt)")
    ap.add_argument("--pred", default="/home/yifan/studium/master_thesis/VGGT-X/data/MAD/11Pig/transforms_converted_openGL.json")
    ap.add_argument("--gt", default="/home/yifan/studium/master_thesis/VGGT-X/data/MAD/11Pig/transforms_train.json")
    ap.add_argument("--align", action="store_true", help="Umeyama-align pred to GT (centers)")
    args = ap.parse_args()

    T_pred = load_tf(args.pred)
    T_gt   = load_tf(args.gt)

    keys = sorted(set(T_pred.keys()) & set(T_gt.keys()))
    if not keys:
        raise SystemExit("No overlapping file_path between the two transforms.json")

    C_pred = np.stack([T_pred[k][:3,3] for k in keys], 0)
    R_pred = np.stack([T_pred[k][:3,:3] for k in keys], 0)
    C_gt   = np.stack([T_gt[k][:3,3]   for k in keys], 0)
    R_gt   = np.stack([T_gt[k][:3,:3]  for k in keys], 0)

    # 可选：对齐预测相机中心到 GT
    if args.align:
        s, Rg, tg = umeyama_align(C_pred, C_gt)
        C_pred = (s * (C_pred @ Rg.T)) + tg
        R_pred = np.einsum('ij,njk->nik', Rg, R_pred)
        print(f"[Align] scale={s:.6f}\n[Align] Rg=\n{Rg}\n[Align] t={tg}")

    # 误差（逐帧）
    t_err = np.linalg.norm(C_pred - C_gt, axis=1)
    r_err = np.array([rot_err_deg(R_pred[i], R_gt[i]) for i in range(len(keys))])
    print(f"Trans Error: mean={t_err.mean():.6g}, median={np.median(t_err):.6g}, max={t_err.max():.6g}")
    print(f"Rot   Error: mean={r_err.mean():.6g} deg, median={np.median(r_err):.6g} deg, max={r_err.max():.6g} deg")
    print(f"Pairs compared: {len(keys)}")

    # 可视化
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(C_gt[:,0],   C_gt[:,1],   C_gt[:,2],   s=18, label="GT",   marker='o')
    ax.scatter(C_pred[:,0], C_pred[:,1], C_pred[:,2], s=14, label="Pred", marker='^')

    # 画配对连线方便看偏差
    for a,b in zip(C_gt, C_pred):
        ax.plot([a[0],b[0]], [a[1],b[1]], [a[2],b[2]], linewidth=0.5)

    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title("Camera centers: GT (o) vs Pred (^) " + ("[aligned]" if args.align else ""))
    ax.legend(); plt.tight_layout(); plt.show()

if __name__ == "__main__":
    main()
