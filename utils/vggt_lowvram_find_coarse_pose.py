#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import json
import math
import random
import argparse
from tqdm import tqdm
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from factory.vggt_low_vram.vggt.models.vggt import VGGT
from factory.vggt_low_vram.vggt.utils.load_fn import load_and_preprocess_images_square
from factory.vggt_low_vram.vggt.utils.pose_enc import pose_encoding_to_extri_intri

# for deterministic behavior
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False


# -------------------------
# VGGT inference
# -------------------------
@torch.no_grad()
def run_VGGT(model, images, device, dtype, resolution=518):
    """
    images: (N, 3, H, W) torch float in [0,1] or [0,255] depending on loader
    returns:
      extrinsic: (N, 4, 4) or (N, 3, 4) in OpenCV convention (camera from world) == w2c
      intrinsic: (N, 3, 3)
      depth_map/conf: (N, H, W) or similar (depends on model)
    """
    assert images.ndim == 4 and images.shape[1] == 3

    images = F.interpolate(images, size=(resolution, resolution), mode="bilinear", align_corners=False)
    images = images.to(device=device, dtype=dtype)

    images = images[None]  # add batch dim
    aggregated_tokens_list, ps_idx = model.aggregator(images, verbose=False)

    pose_enc = model.camera_head(aggregated_tokens_list)[-1]
    extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])  # OpenCV w2c

    depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)

    extrinsic = extrinsic.squeeze(0).cpu().numpy()
    intrinsic = intrinsic.squeeze(0).cpu().numpy()
    depth_map = depth_map.squeeze(0).cpu().numpy()
    depth_conf = depth_conf.squeeze(0).cpu().numpy()
    return extrinsic, intrinsic, depth_map, depth_conf


def opencv_to_opengl(T_c2w: np.ndarray) -> np.ndarray:
    # OpenCV camera coords -> OpenGL/Blender (flip y/z)
    fix = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float64)
    return T_c2w @ fix


def write_transforms_json_from_vggt(
    extrinsic_w2c: np.ndarray,   # (N,4,4) or (N,3,4)
    intrinsic: np.ndarray,       # (N,3,3) or (3,3)
    image_paths: list,           # length N, full paths (recommended)
    original_coords: np.ndarray, # (N,4) typically [top_left_x, top_left_y, W, H]
    img_size: int,               # vggt resolution (518)
    out_path: str,
):
    extrinsic_w2c = np.asarray(extrinsic_w2c)
    intrinsic = np.asarray(intrinsic)

    # normalize extrinsic to (N,4,4)
    if extrinsic_w2c.ndim == 3 and extrinsic_w2c.shape[1:] == (3, 4):
        N = extrinsic_w2c.shape[0]
        tmp = np.zeros((N, 4, 4), dtype=np.float64)
        tmp[:, :3, :4] = extrinsic_w2c
        tmp[:, 3, 3] = 1.0
        extrinsic_w2c = tmp
    elif extrinsic_w2c.ndim == 3 and extrinsic_w2c.shape[1:] == (4, 4):
        pass
    else:
        raise ValueError(f"Unsupported extrinsic shape: {extrinsic_w2c.shape}")

    N = extrinsic_w2c.shape[0]
    if intrinsic.ndim == 2:
        intrinsic = np.repeat(intrinsic[None, ...], N, axis=0)
    elif intrinsic.ndim == 3 and intrinsic.shape[0] == N:
        pass
    else:
        raise ValueError(f"Unsupported intrinsic shape: {intrinsic.shape}")

    if len(image_paths) != N:
        raise ValueError(f"len(image_paths)={len(image_paths)} != N={N}")
    if original_coords.shape[0] != N:
        raise ValueError(f"original_coords has {original_coords.shape[0]} entries but N={N}")

    frames = []

    # compute camera_angle_x using first frame after rescale-to-original
    real_wh0 = original_coords[0, -2:].astype(np.float64)  # (W,H)
    resize_ratio0 = max(real_wh0) / float(img_size)

    K0 = intrinsic[0].astype(np.float64).copy()
    K0[:2, :] *= resize_ratio0
    K0[0, 2] = real_wh0[0] / 2.0
    K0[1, 2] = real_wh0[1] / 2.0
    fx0 = float(K0[0, 0])
    camera_angle_x = 2.0 * math.atan(float(real_wh0[0]) / (2.0 * fx0))

    for i in range(N):
        w2c = extrinsic_w2c[i].astype(np.float64)
        c2w = np.linalg.inv(w2c)
        c2w = opencv_to_opengl(c2w)

        real_wh = original_coords[i, -2:].astype(np.float64)  # (W,H)
        resize_ratio = max(real_wh) / float(img_size)

        K = intrinsic[i].astype(np.float64).copy()
        K[:2, :] *= resize_ratio
        K[0, 2] = real_wh[0] / 2.0
        K[1, 2] = real_wh[1] / 2.0

        file_path = image_paths[i].replace("\\", "/")

        frames.append({
            "file_path": file_path,
            "transform_matrix": c2w.tolist(),
            "fl_x": float(K[0, 0]), "fl_y": float(K[1, 1]),
            "cx": float(K[0, 2]), "cy": float(K[1, 2]),
            "w": int(real_wh[0]), "h": int(real_wh[1]),
            "camera_model": "PINHOLE",
        })

    transforms = {
        "camera_angle_x": float(camera_angle_x),
        "frames": frames,
    }

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(transforms, f, indent=2)
    print(f"[OK] wrote transforms.json: {out_path}  (#frames={N})")


# -------------------------
# Packing utilities
# -------------------------
def list_images_sorted(folder: str):
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    paths = []
    for p in glob.glob(os.path.join(folder, "*")):
        if os.path.splitext(p.lower())[1] in exts:
            paths.append(p)
    paths.sort()
    return paths


def safe_stem(p: str) -> str:
    s = Path(p).stem
    # avoid crazy chars in filename
    return "".join([c if c.isalnum() or c in ("-", "_") else "_" for c in s])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_dir", required=True, help="scene dir that contains train images in <scene_dir>/images")
    ap.add_argument("--eval_dir", required=True, help="dir that contains burrs/good/missing/stains")
    ap.add_argument("--out_dir", required=True, help="where to save per-query transforms json")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--img_load_resolution", type=int, default=1024)
    ap.add_argument("--vggt_resolution", type=int, default=518)
    ap.add_argument("--save_jsonl", action="store_true", help="also append query poses to a merged jsonl file")
    args = ap.parse_args()

    # seed
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8) else torch.float16
    print(f"device={device}, dtype={dtype}")

    # load model
    model = VGGT()
    _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
    model.eval()
    model = model.to(device=device, dtype=dtype)
    print("[OK] model loaded")

    # train images (fixed set)
    train_dir = os.path.join(args.scene_dir, "images")
    train_paths = list_images_sorted(train_dir)
    if not train_paths:
        raise RuntimeError(f"No train images found in {train_dir}")
    print(f"[OK] train images: {len(train_paths)}")

    # preload train tensors ONCE (fast!)
    train_imgs, train_coords_t = load_and_preprocess_images_square(train_paths, args.img_load_resolution)
    train_coords = train_coords_t.cpu().numpy() if torch.is_tensor(train_coords_t) else train_coords_t

    # forward once for train images
    train_extri, train_intri, _, _ = run_VGGT(model, train_imgs, device, dtype, args.vggt_resolution)
    print(f"[OK] train poses computed: {train_extri.shape[0]} images")
    out_name = f"transforms_anomaly_free_poses.json"
    out_path = os.path.join(args.out_dir, out_name)
    write_transforms_json_from_vggt(
        extrinsic_w2c=train_extri,
        intrinsic=train_intri,
        image_paths=train_paths,
        original_coords=train_coords_t.cpu().numpy(),
        img_size=args.vggt_resolution,
        out_path=out_path,
    )


    # queries
    subsets = ["Burrs", "good", "Missing", "Stains"]
    all_queries = []
    for s in subsets:
        d = os.path.join(args.eval_dir, s)
        if os.path.isdir(d):
            q = list_images_sorted(d)
            all_queries += [(s, p) for p in q]
    if not all_queries:
        raise RuntimeError(f"No query images found under {args.eval_dir}/{{burrs,good,missing,stains}}")
    print(f"[OK] total queries: {len(all_queries)}")

    os.makedirs(args.out_dir, exist_ok=True)

    jsonl_path = os.path.join(args.out_dir, "query_poses_merged.jsonl")
    if args.save_jsonl and os.path.exists(jsonl_path):
        os.remove(jsonl_path)

    # loop
    for idx, (subset, qpath) in enumerate(
        tqdm(all_queries, desc="Processing queries", unit="img")
    ):
        # load single query
        q_imgs, q_coords_t = load_and_preprocess_images_square([qpath], args.img_load_resolution)
        q_coords = q_coords_t.cpu().numpy() if torch.is_tensor(q_coords_t) else q_coords_t

        # pack = train + query (no file copy)
        packed_imgs = torch.cat([train_imgs, q_imgs], dim=0)
        packed_coords = np.concatenate([train_coords, q_coords], axis=0)
        packed_paths = train_paths + [qpath]

        # run vggt
        extri, intri, depth, conf = run_VGGT(model, packed_imgs, device, dtype, args.vggt_resolution)

        # write per-query transforms (train + this query)
        out_name = f"transforms_{subset}_{idx:05d}_{safe_stem(qpath)}.json"
        out_path = os.path.join(args.out_dir, "verbose_transforms_file", out_name)
        write_transforms_json_from_vggt(
            extrinsic_w2c=extri,
            intrinsic=intri,
            image_paths=packed_paths,
            original_coords=packed_coords,
            img_size=args.vggt_resolution,
            out_path=out_path,
        )

        # (optional) also append just the query pose into a merged jsonl
        if args.save_jsonl:
            # query is the last frame
            q_c2w_opengl = np.linalg.inv(to_4x4(extri)[-1])
            q_c2w_opengl = opencv_to_opengl(q_c2w_opengl)
            rec = {
                "subset": subset,
                "query_path": qpath.replace("\\", "/"),
                "transforms_json": out_path.replace("\\", "/"),
                "query_transform_matrix": q_c2w_opengl.tolist(),
            }
            with open(jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")

        if (idx + 1) % 10 == 0 or idx == 0:
            print(f"[{idx+1}/{len(all_queries)}] done")

    print("[DONE] all queries processed")


def to_4x4(extri):
    extri = np.asarray(extri)
    if extri.ndim == 3 and extri.shape[1:] == (3, 4):
        N = extri.shape[0]
        T = np.zeros((N, 4, 4), dtype=np.float64)
        T[:, :3, :4] = extri
        T[:, 3, 3] = 1.0
        return T
    return extri.astype(np.float64)


if __name__ == "__main__":
    main()
