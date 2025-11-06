#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import argparse
import numpy as np

def load_transforms(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    frames = data.get("frames", [])
    mats = [np.array(fr["transform_matrix"], dtype=float) for fr in frames]
    return data, frames, mats

def save_transforms(path, data, frames, mats_new, meta):
    # 写回新的矩阵
    for fr, M in zip(frames, mats_new):
        fr["transform_matrix"] = M.tolist()
    # 记录归一化信息，方便溯源
    data["normalization"] = meta
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"[OK] wrote normalized transforms to: {path}")

def choose_center(C, center_mode="mean"):
    if center_mode == "mean":
        return C.mean(axis=0)
    elif center_mode == "median":
        return np.median(C, axis=0)
    elif center_mode == "zero":
        return np.zeros(3)
    else:
        raise ValueError(f"Unknown center_mode: {center_mode}")

def choose_scale(C0, scale_mode="max", percentile=90, target=1.0):
    # C0 是中心化后的相机中心
    d = np.linalg.norm(C0, axis=1) + 1e-12
    if scale_mode == "max":
        base = d.max()
    elif scale_mode == "mean":
        base = d.mean()
    elif scale_mode == "median":
        base = np.median(d)
    elif scale_mode == "percentile":
        base = np.percentile(d, percentile)
    else:
        raise ValueError(f"Unknown scale_mode: {scale_mode}")
    if base <= 0:
        # 所有相机都在同一点时，避免除以0
        base = 1.0
    s = float(target) / float(base)
    return s, float(base)

def build_normalizer(center, scale):
    """N = [[sI, -sI*center],[0,1]]"""
    N = np.eye(4, dtype=float)
    N[:3, :3] *= scale
    N[:3, 3] = (-scale * center).ravel()
    return N

def main():
    ap = argparse.ArgumentParser(description="Normalize ground-truth transforms.json by centering & scaling world coords.")
    ap.add_argument("--in_json", required=True, help="Input transforms.json (ground truth).")
    ap.add_argument("--out_json", required=True, help="Output normalized transforms.json.")
    ap.add_argument("--center_mode", default="mean", choices=["mean","median","zero"],
                    help="How to choose center c0 from camera centers. Default: mean.")
    ap.add_argument("--scale_mode", default="max", choices=["max","mean","median","percentile"],
                    help="How to define radius before scaling. Default: max.")
    ap.add_argument("--percentile", type=float, default=90.0,
                    help="Used when scale_mode=percentile. Default: 90.")
    ap.add_argument("--target_radius", type=float, default=1.0,
                    help="After normalization, chosen radius becomes target_radius. Default: 1.0 (unit sphere).")
    args = ap.parse_args()

    data, frames, mats = load_transforms(args.in_json)

    # 提取相机中心（cam2world 的第四列前三个分量）
    C = np.stack([M[:3, 3] for M in mats], axis=0)  # (N,3)

    # 选择中心并中心化
    c0 = choose_center(C, args.center_mode)
    C0 = C - c0[None, :]

    # 选择尺度
    s, base_radius = choose_scale(C0, args.scale_mode, percentile=args.percentile, target=args.target_radius)

    # 构造归一化矩阵 N，并左乘所有 T_c2w
    N = build_normalizer(c0, s)  # 世界 -> 归一化世界
    mats_new = [N @ M for M in mats]

    meta = {
        "center_mode": args.center_mode,
        "scale_mode": args.scale_mode,
        "percentile": args.percentile,
        "target_radius": args.target_radius,
        "computed_center": c0.tolist(),
        "pre_scale_radius": base_radius,
        "scale": s
    }

    save_transforms(args.out_json, data, frames, mats_new, meta)

    # 打印对比信息
    C_new = np.stack([M[:3, 3] for M in mats_new], axis=0)
    d_new = np.linalg.norm(C_new, axis=1)
    print(f"Center used: {c0}")
    print(f"Scale s: {s:.6f} (pre_scale_radius={base_radius:.6f} -> target={args.target_radius})")
    print(f"New radii  mean/median/max: {d_new.mean():.6f} / {np.median(d_new):.6f} / {d_new.max():.6f}")

if __name__ == "__main__":
    main()
