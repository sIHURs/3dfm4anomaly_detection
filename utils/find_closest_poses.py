#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, os
import numpy as np


def load_transforms(path: str):
    with open(path, "r") as f:
        d = json.load(f)
    frames = d.get("frames", [])
    if not frames:
        raise ValueError(f"No frames in {path}")
    return frames


def get_R_C(T: np.ndarray, pose_mode: str):
    """
    Return (R, C) where:
      - R is camera-to-world rotation if pose_mode=c2w, else world-to-camera rotation
      - C is camera center in world coordinates
    """
    T = np.asarray(T, dtype=float)
    R = T[:3, :3]
    t = T[:3, 3]

    if pose_mode == "c2w":
        # T = [R | C]
        C = t
        return R, C
    elif pose_mode == "w2c":
        # T = [R | t], with x_c = R x_w + t
        # camera center C satisfies 0 = R C + t => C = -R^T t
        C = -(R.T @ t)
        return R, C
    else:
        raise ValueError("pose_mode must be c2w or w2c")


def rot_angle_deg(Ra: np.ndarray, Rb: np.ndarray):
    """
    rotation difference angle between two rotations (in degrees)
    """
    R = Ra.T @ Rb
    tr = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(tr)))


def basename_noext(p: str):
    b = os.path.basename(p)
    return os.path.splitext(b)[0]


def main():
    ap = argparse.ArgumentParser("Nearest train poses for each query pose")
    ap.add_argument("--train_json", required=True, type=str, help="transforms_train.json")
    ap.add_argument("--query_json", required=True, type=str, help="query transforms json")
    ap.add_argument("--topk", type=int, default=3, help="number of nearest train poses")
    ap.add_argument("--metric", type=str, default="rot", choices=["rot", "rot_trans"],
                    help="rot: only rotation; rot_trans: rotation + translation")
    ap.add_argument("--pose_mode", type=str, default="c2w", choices=["c2w", "w2c"],
                    help="how to interpret transform_matrix")
    ap.add_argument("--query_prefix", type=str, default="query",
                    help="only treat query frames whose file_path starts with this prefix")
    ap.add_argument("--rot_w", type=float, default=1.0, help="weight for rotation term (deg)")
    ap.add_argument("--trans_w", type=float, default=1.0, help="weight for translation term (normalized)")
    ap.add_argument("--trans_scale", type=float, default=0.0,
                    help="translation normalization scale. 0 means auto (median pairwise train distance)")
    ap.add_argument("--out_json", type=str, default="",
                    help="if set, write matches to this json path")
    args = ap.parse_args()

    train_frames = load_transforms(args.train_json)
    query_frames = load_transforms(args.query_json)

    # -------- prepare train poses --------
    train_list = []
    for fr in train_frames:
        fp = fr.get("file_path", "")
        T = fr.get("transform_matrix", None)
        if T is None:
            continue
        R, C = get_R_C(np.array(T, float), args.pose_mode)
        train_list.append((fp, R, C))

    if len(train_list) < 1:
        raise SystemExit("No valid train poses loaded.")

    # auto trans scale (for rot_trans)
    if args.metric == "rot_trans":
        if args.trans_scale > 0:
            trans_scale = float(args.trans_scale)
        else:
            # robust scale: median distance to train centroid
            Cs = np.stack([x[2] for x in train_list], 0)
            centroid = Cs.mean(0)
            d = np.linalg.norm(Cs - centroid, axis=1)
            trans_scale = float(np.median(d) + 1e-8)
        # print for reproducibility
        print(f"[INFO] trans_scale = {trans_scale:.6g}")

    # -------- iterate query poses (by prefix) --------
    results = {}
    q_count = 0

    for fr in query_frames:
        qfp = fr.get("file_path", "")
        if not qfp.startswith(args.query_prefix):
            continue
        Tq = fr.get("transform_matrix", None)
        if Tq is None:
            continue

        q_count += 1
        Rq, Cq = get_R_C(np.array(Tq, float), args.pose_mode)

        scored = []
        for tfp, Rt, Ct in train_list:
            rdeg = rot_angle_deg(Rt, Rq)

            if args.metric == "rot":
                score = rdeg
                tdist = None
            else:
                tdist_raw = float(np.linalg.norm(Ct - Cq))
                tdist = tdist_raw / trans_scale
                score = args.rot_w * rdeg + args.trans_w * tdist

            scored.append((score, rdeg, tdist, tfp))

        scored.sort(key=lambda x: x[0])
        top = scored[: max(1, args.topk)]

        results[qfp] = [
            {
                "train_file_path": tfp,
                "score": float(score),
                "rot_deg": float(rdeg),
                **({} if tdist is None else {"trans_norm": float(tdist)}),
            }
            for score, rdeg, tdist, tfp in top
        ]

        # print concise
        print(f"\n[QUERY] {qfp}")
        for i, item in enumerate(results[qfp], 1):
            if "trans_norm" in item:
                print(f"  #{i}: score={item['score']:.4f} | rot={item['rot_deg']:.3f}deg | trans_norm={item['trans_norm']:.4f} | {item['train_file_path']}")
            else:
                print(f"  #{i}: score={item['score']:.4f} | rot={item['rot_deg']:.3f}deg | {item['train_file_path']}")

    if q_count == 0:
        print(f"[WARN] No query frames found with prefix '{args.query_prefix}' in {args.query_json}")

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(
                {
                    "train_json": args.train_json,
                    "query_json": args.query_json,
                    "metric": args.metric,
                    "topk": args.topk,
                    "pose_mode": args.pose_mode,
                    "query_prefix": args.query_prefix,
                    "rot_w": args.rot_w,
                    "trans_w": args.trans_w,
                    **({} if args.metric == "rot" else {"trans_scale": trans_scale}),
                    "matches": results,
                },
                f,
                indent=2,
            )
        print(f"\n[Save] {args.out_json}")


if __name__ == "__main__":
    main()
