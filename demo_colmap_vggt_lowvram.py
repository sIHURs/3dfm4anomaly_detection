# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import shutil
import random
import numpy as np
import glob
import os
import copy
import torch
import torch.nn.functional as F

# Configure CUDA settings
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

# for deterministic behavior
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True
# torch.use_deterministic_algorithms(True)


import argparse
from pathlib import Path
import trimesh
import pycolmap
import cv2
import json
from tqdm import tqdm
from datetime import datetime
import math
now = datetime.now()
from utils.time_recorder import SpanTimer

from factory.vggt_low_vram.vggt.models.vggt import VGGT
from factory.vggt_low_vram.vggt.utils.load_fn import load_and_preprocess_images_square
from factory.vggt_low_vram.vggt.utils.pose_enc import pose_encoding_to_extri_intri
from factory.vggt_low_vram.vggt.utils.geometry import unproject_depth_map_to_point_map
from factory.vggt_low_vram.vggt.utils.helper import create_pixel_coordinate_grid, randomly_limit_trues
from factory.vggt_low_vram.vggt.dependency.track_predict import predict_tracks
from factory.vggt_low_vram.vggt.dependency.np_to_pycolmap import batch_np_matrix_to_pycolmap, batch_np_matrix_to_pycolmap_wo_track


# TODO: add support for masks
# TODO: add iterative BA
# TODO: add support for radial distortion, which needs extra_params
# TODO: test with more cases
# TODO: test different camera types

def parse_args():
    parser = argparse.ArgumentParser(description="VGGT Demo")
    parser.add_argument("--scene_dir", type=str, required=True, help="Directory containing the scene images")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save the output reconstruction")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument("--max_points", type=int, default=100000, help="Number of predicted points for colmap")
    parser.add_argument("--use_ba", action="store_true", default=False, help="Use BA for reconstruction")
    ######### BA parameters #########
    parser.add_argument(
        "--max_reproj_error", type=float, default=8.0, help="Maximum reprojection error for reconstruction"
    )
    parser.add_argument("--shared_camera", action="store_true", default=False, help="Use shared camera for all images")
    parser.add_argument("--camera_type", type=str, default="SIMPLE_PINHOLE", help="Camera type for reconstruction")
    parser.add_argument("--vis_thresh", type=float, default=0.2, help="Visibility threshold for tracks")
    parser.add_argument("--query_frame_num", type=int, default=8, help="Number of frames to query")
    parser.add_argument("--max_query_pts", type=int, default=4096, help="Maximum number of query points")
    parser.add_argument(
        "--fine_tracking", action="store_true", default=True, help="Use fine tracking (slower but more accurate)"
    )
    parser.add_argument(
        "--conf_thres_value", type=float, default=5.0, help="Confidence threshold value for depth filtering (wo BA)"
    )
    parser.add_argument(
        "--adjust_folder", action="store_true", default=True, help="adjust the folder structure to match COLMAP format"
    )

    # todo: for sparse view experiments
    parser.add_argument("--eval_dir", help="dir that contains burrs/good/missing/stains")
    parser.add_argument("--test_sparse_view", action="store_true", default=False, help="test with sparse view input")
    parser.add_argument("--query_batch_size", type=int, default=1, help="test with sparse view input")

    # todo: add more parameters for testing & experiments @yifan
    parser.add_argument("--save_depth", action="store_true", default=False, help="Save depth map and confidence map")

    args = parser.parse_args()
    
    # args conditions
    if args.test_sparse_view and args.eval_dir is None:
        parser.error("--eval_dir is required when --test_sparse_view is set")

    return args



''' help functions @yifan '''

def save_vggt_json_w2c(
    extrinsic,
    intrinsic,
    out_path="vggt_extrinsic_intrinsic_w2c.json",
):
    extrinsic = np.asarray(extrinsic)
    intrinsic = np.asarray(intrinsic)

    if intrinsic.ndim == 2:
        intrinsic = intrinsic[None].repeat(extrinsic.shape[0], axis=0)

    frames = []
    for i in range(extrinsic.shape[0]):
        frames.append({
            "frame_id": int(i),
            "extrinsic_w2c": extrinsic[i].tolist(),
            "intrinsic": intrinsic[i].tolist(),
        })

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(frames, f, indent=2)

    # print(f"[OK][{now}] wrote {out_path}")

def save_depth_outputs(depth_map, depth_conf, out_dir, prefix="vggt"):
    os.makedirs(out_dir, exist_ok=True)

    # Tensor → CPU → NumPy
    if torch.is_tensor(depth_map):
        depth_map = depth_map.detach().cpu().numpy()
    if torch.is_tensor(depth_conf):
        depth_conf = depth_conf.detach().cpu().numpy()

    np.save(os.path.join(out_dir, "verbose", f"{prefix}_depth.npy"), depth_map)
    np.save(os.path.join(out_dir, "verbose", f"{prefix}_depth_conf.npy"), depth_conf)

    print(f"[OK][{now}] Saved {prefix} depth_map and depth_conf to {out_dir}")

def save_depth_png(depth, path, vmin=None, vmax=None):
    depth = depth.astype(np.float32)

    if vmin is None:
        vmin = np.percentile(depth, 2)
    if vmax is None:
        vmax = np.percentile(depth, 98)

    depth_norm = (depth - vmin) / (vmax - vmin + 1e-6)
    depth_norm = np.clip(depth_norm, 0, 1)

    depth_uint8 = (depth_norm * 255).astype(np.uint8)
    cv2.imwrite(path, depth_uint8)


def topk_points_from_conf(c_np, k=3):
    H, W = c_np.shape
    flat = c_np.reshape(-1)

    # Top-K indices (descending)
    idx = np.argpartition(-flat, k)[:k]
    idx = idx[np.argsort(-flat[idx])]  # sort by value

    ys, xs = np.unravel_index(idx, (H, W))
    scores = flat[idx]

    return list(zip(xs, ys, scores))


def _apply_nms_mask(c: np.ndarray, x: int, y: int, min_dist: int,
                    shape: str = "square", value: float = 0.0) -> None:
    """
    Apply a non-circular NMS suppression mask centered at (x, y).
    shape: "square" | "diamond" | "cross"
    """
    h, w = c.shape
    r = int(min_dist)

    y0 = max(0, y - r)
    y1 = min(h, y + r + 1)
    x0 = max(0, x - r)
    x1 = min(w, x + r + 1)

    if shape == "square":
        c[y0:y1, x0:x1] = value

    elif shape == "diamond":
        # |dx| + |dy| <= r
        for yy in range(y0, y1):
            dy = abs(yy - y)
            rem = r - dy
            if rem < 0:
                continue
            xx0 = max(x0, x - rem)
            xx1 = min(x1, x + rem + 1)
            c[yy, xx0:xx1] = value

    elif shape == "cross":
        # Horizontal + vertical lines with radius r
        c[y, x0:x1] = value
        c[y0:y1, x] = value

    else:
        raise ValueError(f"Unknown shape='{shape}'. Use 'square', 'diamond', or 'cross'.")


def topk_points_nms(
    c_np: np.ndarray,
    k: int = 3,
    min_dist: int = 20,
    shape: str = "square",
    weight_mode: str = "linear",
    conf_pow: float = 1.0,
    eps: float = 1e-8,
    stop_if_nonpositive: bool = True,
    avg_score_mode: str = "sum",   # "sum" | "mean"
):
    """
    Pick top-k peaks with NMS and append a confidence-weighted average point
    in the SAME (x, y, score) format as other points.

    Returns:
        points: [(x, y, conf), ..., (x_mean, y_mean, avg_score)]
                The last one is the weighted average point.
    """
    if c_np.ndim != 2:
        raise ValueError(f"c_np must be 2D, got shape {c_np.shape}")

    c = c_np.copy()
    points = []

    for _ in range(int(k)):
        idx = int(np.argmax(c))
        score = float(c.flat[idx])

        if stop_if_nonpositive and ((not np.isfinite(score)) or (score <= 0.0)):
            break

        y, x = np.unravel_index(idx, c.shape)
        x = int(x)
        y = int(y)

        points.append((float(x), float(y), float(score)))
        _apply_nms_mask(c, x, y, min_dist=min_dist, shape=shape, value=0.0)

    # Append confidence-weighted average point (same tuple format)
    if len(points) > 0:
        xs = np.array([p[0] for p in points], dtype=np.float64)
        ys = np.array([p[1] for p in points], dtype=np.float64)
        cs = np.array([p[2] for p in points], dtype=np.float64)

        if weight_mode == "linear":
            w = np.maximum(cs, 0.0)
            w_sum = float(w.sum())
            w = (np.ones_like(w) / len(w)) if (w_sum <= eps) else (w / (w_sum + eps))

        elif weight_mode == "softmax":
            z = cs - cs.max()
            w = np.exp(z)
            w = w / (float(w.sum()) + eps)

        elif weight_mode == "power":
            w = np.maximum(cs, 0.0) ** float(conf_pow)
            w_sum = float(w.sum())
            w = (np.ones_like(w) / len(w)) if (w_sum <= eps) else (w / (w_sum + eps))

        else:
            raise ValueError("weight_mode must be 'linear', 'softmax', or 'power'.")

        x_mean = float((xs * w).sum())
        y_mean = float((ys * w).sum())

        conf_mean = float(cs.mean())
        conf_sum  = float(cs.sum())

        if avg_score_mode == "sum":
            avg_score = conf_sum
        elif avg_score_mode == "mean":
            avg_score = conf_mean
        else:
            raise ValueError("avg_score_mode must be 'sum' or 'mean'.")

        # IMPORTANT: same (x, y, score) format
        points.append((x_mean, y_mean, avg_score))

    return points

def draw_points_on_image(
    img_bgr_uint8,
    points,
    color_point=(0, 0, 255),   # red
    color_peak=(255, 0, 0),    # blue (BGR)
):
    """
    img_bgr_uint8: (H, W, 3) uint8 BGR
    points: [(x, y, score), ...]
    """
    vis = img_bgr_uint8.copy()
    H, W = vis.shape[:2]

    if len(points) == 0:
        return vis

    # -------- find highest-score point --------
    peak_idx = max(range(len(points)), key=lambda i: points[i][2])

    for i, (x, y, score) in enumerate(points):
        x_i = int(np.clip(round(x), 0, W - 1))
        y_i = int(np.clip(round(y), 0, H - 1))

        # ===== highest-score point =====
        if i == peak_idx:
            size = 8
            thickness = 3

            # draw diamond ◇
            pts = np.array([
                [x_i, y_i - size],
                [x_i + size, y_i],
                [x_i, y_i + size],
                [x_i - size, y_i],
            ], np.int32).reshape((-1, 1, 2))

            cv2.polylines(
                vis,
                [pts],
                isClosed=True,
                color=color_peak,
                thickness=thickness,
            )

            cv2.putText(
                vis,
                f"max:{score:.2f}",
                (x_i + 8, max(0, y_i - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color_peak,
                2,
                cv2.LINE_AA,
            )

        # ===== other points =====
        else:
            cv2.circle(vis, (x_i, y_i), 7, (0, 0, 0), -1)
            cv2.circle(vis, (x_i, y_i), 5, color_point, -1)

            cv2.putText(
                vis,
                f"{i+1}:{score:.2f}",
                (x_i + 8, max(0, y_i - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color_point,
                2,
                cv2.LINE_AA,
            )

    return vis

def map_points_depth_to_original(points, original_coords_i, depth_size, square_size):
    """
    points: [(x_d, y_d, score), ...] in depth/conf coords (depth_size)
    original_coords_i: [x1, y1, x2, y2, W0, H0] in square_size coords
    returns: [(x0, y0, score), ...] in original image coords (W0, H0)
    """
    x1, y1, x2, y2, W0, H0 = original_coords_i
    W0, H0 = float(W0), float(H0)

    sx = square_size / float(depth_size)

    out = []
    for x_d, y_d, s in points:
        # depth -> square
        x_sq = x_d * sx
        y_sq = y_d * sx

        # square -> original
        x0 = (x_sq - x1) * (W0 / (x2 - x1 + 1e-6))
        y0 = (y_sq - y1) * (H0 / (y2 - y1 + 1e-6))

        x0 = float(np.clip(x0, 0, W0 - 1))
        y0 = float(np.clip(y0, 0, H0 - 1))
        out.append((x0, y0, s))
    return out

def restructure_scene_dir(args):
    """
    Original folder structure:
      scene_dir/
        images/
        sparse/
          0/
            images.bin
            cameras.bin
            points3D.bin

    Target folder structure:
      scene_dir/
        input/
        distorted/
          sparse/
            0/
              images.bin
              cameras.bin
              points3D.bin
    """
    output_dir = args.output_dir
    images_dir = os.path.join(output_dir, "images")
    sparse_dir = os.path.join(output_dir, "sparse")

    # Target paths
    input_dir = os.path.join(output_dir, "input")
    new_sparse_dir = os.path.join(output_dir, "distorted", "sparse", "0")
    os.makedirs(new_sparse_dir, exist_ok=True)

    # 1️⃣ Rename "images" to "input"
    if os.path.exists(images_dir):
        if os.path.exists(input_dir):
            print(f"⚠️[{now}] Target directory already exists: {input_dir}, skipping rename.")
        else:
            shutil.move(images_dir, input_dir)
            print(f"✅[{now}] Renamed 'images' to 'input'")

    # 2️⃣ Move files from "sparse/0" to "distorted/sparse/0"
    if os.path.exists(sparse_dir):
        for file_name in os.listdir(sparse_dir):
            src = os.path.join(sparse_dir, file_name)
            dst = os.path.join(new_sparse_dir, file_name)
            if os.path.isfile(src):
                shutil.move(src, dst)
        print(f"✅[{now}]Moved contents of 'sparse/0' to {new_sparse_dir}")

    # 3️⃣ Delete the old "sparse" directory
    old_sparse_root = os.path.join(output_dir, "sparse")
    if os.path.exists(old_sparse_root):
        shutil.rmtree(old_sparse_root)
        print(f"🗑️[{now}] Removed old 'sparse' folder")

    print(f"🎯[{now}] Folder structure successfully adjusted: {output_dir}")
    

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
        N = extrinsic_w2c.shape[0]
    else:
        raise ValueError(f"Unsupported extrinsic shape: {extrinsic_w2c.shape}")

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

    # --- compute camera_angle_x/y using first frame after rescale-to-original ---
    real_wh0 = original_coords[0, -2:].astype(np.float64)  # (W,H)
    resize_ratio0 = max(real_wh0) / float(img_size)

    K0 = intrinsic[0].astype(np.float64).copy()
    K0[:2, :] *= resize_ratio0
    K0[0, 2] = real_wh0[0] / 2.0
    K0[1, 2] = real_wh0[1] / 2.0

    fx0 = float(K0[0, 0])
    fy0 = float(K0[1, 1])

    camera_angle_x = 2.0 * math.atan(float(real_wh0[0]) / (2.0 * fx0))
    camera_angle_y = 2.0 * math.atan(float(real_wh0[1]) / (2.0 * fy0))

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
        "camera_angle_y": float(camera_angle_y),  # <-- added
        "frames": frames,
    }

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(transforms, f, indent=2)
    # print(f"[OK][{now}] wrote transforms.json: {out_path}  (#frames={N})")


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

def to_4x4(extri):
    extri = np.asarray(extri)
    if extri.ndim == 3 and extri.shape[1:] == (3, 4):
        N = extri.shape[0]
        T = np.zeros((N, 4, 4), dtype=np.float64)
        T[:, :3, :4] = extri
        T[:, 3, 3] = 1.0
        return T
    return extri.astype(np.float64)


def run_VGGT(model, images, device, dtype, resolution=518):
    # images: [B, 3, H, W]

    assert len(images.shape) == 4
    assert images.shape[1] == 3

    # hard-coded to use 518 for VGGT
    images = F.interpolate(images, size=(resolution, resolution), mode="bilinear", align_corners=False)
    images = images.to(device, dtype)

    with torch.no_grad():
        images = images[None]  # add batch dimension
        aggregated_tokens_list, ps_idx = model.aggregator(images, verbose=True)

        # Predict Cameras
        pose_enc = model.camera_head(aggregated_tokens_list)[-1]
        # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
        # Predict Depth Maps
        depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)

    extrinsic = extrinsic.squeeze(0).cpu().numpy()
    intrinsic = intrinsic.squeeze(0).cpu().numpy()
    depth_map = depth_map.squeeze(0).cpu().numpy()
    depth_conf = depth_conf.squeeze(0).cpu().numpy()
    return extrinsic, intrinsic, depth_map, depth_conf


def demo_fn(args):
    # Print configuration
    if args.output_dir is None:
        args.output_dir = os.path.join(args.scene_dir)

    print("[{now}] Arguments:", vars(args))

    # set time recorder
    tm = SpanTimer()

    # Set seed for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)  # for multi-GPU
    print(f"Setting seed as: {args.seed}")

    # Set device and dtype
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Using dtype: {dtype}")

    # Run VGGT for camera and depth estimation
    tm.mark("before_load_images")
    model = VGGT()
    _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
    model.eval()
    model = model.to(dtype=dtype, device=device)
    tm.mark("after_load_images")
    print(f"-- Model loaded --")

    # Get image paths and preprocess them
    image_dir = os.path.join(args.scene_dir, "images")
    image_path_list = list_images_sorted(image_dir)
    if len(image_path_list) == 0:
        raise ValueError(f"No images found in {image_dir}")

    print(f"[OK][{now}] train images: {len(image_path_list)} from {image_dir}")
    base_image_path_list = [os.path.basename(path) for path in image_path_list]

    # print("[DEBUG] image_path_list: ", image_path_list)
    # print("[DEBUG] base_image_path_list: ", base_image_path_list)

    # Load images and original coordinates
    # Load Image in 1024, while running VGGT with 518
    vggt_fixed_resolution = 518
    img_load_resolution = 1024

    images, original_coords = load_and_preprocess_images_square(image_path_list, img_load_resolution)
    print(f"[OK][{now}] Loaded {len(images)} images from {image_dir}")

    # Run VGGT to estimate camera and depth
    # Run with 518x518 images
    tm.mark("before_run_vggt")
    extrinsic, intrinsic, depth_map, depth_conf = run_VGGT(model, images, device, dtype, vggt_fixed_resolution)
    tm.mark("after_run_vggt")
    points_3d = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)
    # images = images.float()

    # save as blender - transforms.json
    print(f"[OK][{now}] train poses computed: {extrinsic.shape[0]} images")
    out_name = f"transforms_anomaly_free_poses_uncentered.json"
    out_path = os.path.join(args.output_dir, out_name)
    write_transforms_json_from_vggt(
        extrinsic_w2c=extrinsic,
        intrinsic=intrinsic,
        image_paths=base_image_path_list,
        original_coords=original_coords.cpu().numpy() if torch.is_tensor(original_coords) else original_coords,
        img_size=vggt_fixed_resolution,
        out_path=out_path,
    )

    if args.test_sparse_view:
        tm.mark("before_find_query_poses")
        print(f"TESTING SPARSE VIEW INPUT")
        print(f"[OK][{now}] Preparing query images from {args.eval_dir}")

        subsets = ["Burrs", "good", "Missing", "Stains", "scratched", "stained", "squeezed"]
        wanted = {s.lower(): s for s in subsets}

        existing_dirs = {
            name.lower(): name
            for name in os.listdir(args.eval_dir)
            if os.path.isdir(os.path.join(args.eval_dir, name))
        }

        all_queries = []
        for s in subsets:
            key = s.lower()
            if key not in existing_dirs:
                continue
            real_name = existing_dirs[key]
            d = os.path.join(args.eval_dir, real_name)

            q = list_images_sorted(d)
            all_queries += [(s, p) for p in q]

        if not all_queries:
            raise RuntimeError(
                f"No query images found under {args.eval_dir}/{{burrs,good,missing,stains}}"
            )
        print(f"[OK][{now}] total queries: {len(all_queries)}")

        # ------------------------------------------------------------
        # 1) Keep only first 10 train images (and aligned metadata)
        # ------------------------------------------------------------
        keep_train = 10
        images_10 = images[:keep_train]  # torch tensor (T, C, H, W)
        original_coords_10 = original_coords[:keep_train]  # numpy or tensor
        image_path_list_10 = image_path_list[:keep_train]  # list[str]

        # ------------------------------------------------------------
        # 2) Load ALL query images once
        # ------------------------------------------------------------
        qpaths_all = [p for (_, p) in all_queries]
        q_imgs, q_coords_t = load_and_preprocess_images_square(qpaths_all, img_load_resolution)
        q_coords = q_coords_t.cpu().numpy() if torch.is_tensor(q_coords_t) else q_coords_t

        # ------------------------------------------------------------
        # 3) Pack: 10 train + ALL queries, run VGGT ONCE
        # ------------------------------------------------------------
        packed_imgs = torch.cat([images_10, q_imgs], dim=0)
        packed_coords = np.concatenate([original_coords_10, q_coords], axis=0)
        packed_paths = image_path_list_10 + qpaths_all
        packed_paths_name = [os.path.basename(p) for p in packed_paths]

        print(f"[OK][{now}] Packed {len(image_path_list_10)} train + {len(qpaths_all)} query = {len(packed_paths)} total frames")

        extri, intri, depth_map_query, depth_conf_query = run_VGGT(model, packed_imgs, device, dtype, vggt_fixed_resolution)

        out_path = os.path.join(args.output_dir, "transforms_query_poses_uncentered.json")
        write_transforms_json_from_vggt(
            extrinsic_w2c=extri,
            intrinsic=intri,
            image_paths=packed_paths_name,
            original_coords=packed_coords,
            img_size=vggt_fixed_resolution,
            out_path=out_path,
        )

        print(f"[DONE][{now}] all queries processed")
        tm.mark("after_find_query_poses")

        if args.save_depth:
            os.makedirs(args.output_dir, exist_ok=True)
            depth_map_dir = os.path.join(args.output_dir, "verbose", "depth_map_query")
            depth_conf_dir = os.path.join(args.output_dir, "verbose", "depth_conf_map_query")
            os.makedirs(depth_map_dir, exist_ok=True)
            os.makedirs(depth_conf_dir, exist_ok=True)
            
            # save_depth_outputs(depth_map_query, depth_conf_query, out_dir=args.output_dir, prefix="vggt_query")

            for i in range(depth_map_query.shape[0]):
                import matplotlib.pyplot as plt

                base_name = os.path.splitext(packed_paths_name[i])[0]

                save_depth_png(
                    depth_map_query[i],
                    os.path.join(
                        args.output_dir,
                        "verbose",
                        "depth_map_query",
                        f"depth_{base_name}.png",
                    )
                )

                c = depth_conf_query[i]
                c_np = (
                    c.detach().float().cpu().numpy()
                    if torch.is_tensor(c)
                    else c.astype(np.float32)
                )

                # percentile for robust visualization
                vmin = np.percentile(c_np, 5)
                vmax = np.percentile(c_np, 95)

                # 1) clip
                c_clip = np.clip(c_np, vmin, vmax)

                # 2) normalize to [0,255]
                c_norm = (c_clip - vmin) / (vmax - vmin + 1e-6)
                c_uint8 = (c_norm * 255).astype(np.uint8)

                # 3) apply colormap (JET / TURBO / INFERNO 都可以)
                conf_heatmap = cv2.applyColorMap(c_uint8, cv2.COLORMAP_TURBO)
                # alternatives:
                # cv2.COLORMAP_JET
                # cv2.COLORMAP_INFERNO
                # cv2.COLORMAP_VIRIDIS

                # 4) save (BGR already)
                out_path = os.path.join(
                    args.output_dir,
                    "verbose",
                    "depth_conf_map_query",
                    f"depth_conf_query_heatmap_{base_name}.png",
                )
                cv2.imwrite(out_path, conf_heatmap)

    if args.save_depth:
        os.makedirs(args.output_dir, exist_ok=True)
        depth_map_dir = os.path.join(args.output_dir, "verbose", "depth_map")
        depth_conf_dir = os.path.join(args.output_dir, "verbose", "depth_conf_map")
        # conf_points_dir = os.path.join(args.output_dir, "verbose", "conf_points")
        # conf_hist_dir = os.path.join(args.output_dir, "verbose", "depth_conf_hist")
        os.makedirs(depth_map_dir, exist_ok=True)
        os.makedirs(depth_conf_dir, exist_ok=True)
        # os.makedirs(conf_points_dir, exist_ok=True)
        # os.makedirs(conf_hist_dir, exist_ok=True)
        
        # save_depth_outputs(depth_map, depth_conf, out_dir=args.output_dir, prefix="vggt")

        for i in range(depth_map.shape[0]):
            import matplotlib.pyplot as plt

            base_name = os.path.splitext(base_image_path_list[i])[0]

            save_depth_png(
                depth_map[i],
                os.path.join(
                    args.output_dir,
                    "verbose",
                    "depth_map",
                    f"depth_{base_name}.png",
                )
            )

            c = depth_conf[i]
            c_np = (
                c.detach().float().cpu().numpy()
                if torch.is_tensor(c)
                else c.astype(np.float32)
            )

            # percentile for robust visualization
            vmin = np.percentile(c_np, 5)
            vmax = np.percentile(c_np, 95)

            # 1) clip
            c_clip = np.clip(c_np, vmin, vmax)

            # 2) normalize to [0,255]
            c_norm = (c_clip - vmin) / (vmax - vmin + 1e-6)
            c_uint8 = (c_norm * 255).astype(np.uint8)

            # 3) apply colormap (JET / TURBO / INFERNO 都可以)
            conf_heatmap = cv2.applyColorMap(c_uint8, cv2.COLORMAP_TURBO)
            # alternatives:
            # cv2.COLORMAP_JET
            # cv2.COLORMAP_INFERNO
            # cv2.COLORMAP_VIRIDIS

            # 4) save (BGR already)
            out_path = os.path.join(
                args.output_dir,
                "verbose",
                "depth_conf_map",
                f"depth_conf_heatmap_{base_name}.png",
            )
            cv2.imwrite(out_path, conf_heatmap)

            # # if normalize
            # c_proc = np.clip(c_np, vmin, vmax)
            # c_proc = (c_proc - vmin) / (vmax - vmin + 1e-6)

            # # if mask
            # depth_i = depth_map[i].detach().float().cpu().numpy() if torch.is_tensor(depth_map[i]) else depth_map[i]
            # mask = np.isfinite(depth_i) & (depth_i > 0)
            # vals = c_proc[mask].reshape(-1)
            
            # vals = c_np.reshape(-1)

            # fig = plt.figure()
            # plt.hist(vals, bins=50)
            # plt.title(f"depth_conf histogram: {base_name}\nclip p5={vmin:.3g}, p95={vmax:.3g}")
            # plt.xlabel("conf (clipped & normalized to [0,1])")
            # plt.ylabel("count")
            # hist_path = os.path.join(conf_hist_dir, f"depth_conf_hist_{base_name}.png")
            # plt.tight_layout()
            # plt.savefig(hist_path, dpi=150)
            # plt.close(fig)

            # pts_depth = topk_points_nms(c_np, k=10, min_dist=15, shape="diamond")
            # depth_size = int(c_np.shape[-1])         # e.g. 518
            # square_size = int(images.shape[-1])      # e.g. 1024

            # pts_orig = map_points_depth_to_original(
            #     pts_depth,
            #     original_coords[i].detach().cpu().numpy(),
            #     depth_size=depth_size,
            #     square_size=square_size,
            # )

            # orig_path = image_path_list[i]
            # orig_bgr = cv2.imread(orig_path, cv2.IMREAD_COLOR)
            # if orig_bgr is None:
            #     raise ValueError(f"Failed to read image: {orig_path}")
            
            # conf_vis = draw_points_on_image(orig_bgr, pts_orig)

            # out_path = os.path.join(
            #     args.output_dir,
            #     "verbose",
            #     "conf_points",
            #     f"depth_conf_top3_on_orig_{base_name}.png",
            # )

            # ok = cv2.imwrite(out_path, conf_vis)
            # if not ok:
            #     raise IOError(
            #         f"cv2.imwrite failed: {out_path}, "
            #         f"shape={conf_vis.shape}, dtype={conf_vis.dtype}"
            #     )

    del model  # free memory
    torch.cuda.empty_cache()

    images = images.to(device, dtype)
    original_coords = original_coords.to(device)

    print(f"[OK] Converting to COLMAP format and saving reconstruction")

    if args.use_ba:
        image_size = np.array(images.shape[-2:])
        scale = img_load_resolution / vggt_fixed_resolution
        shared_camera = args.shared_camera

        # TODO: use VGGT tracker
        with torch.inference_mode():
            # Predicting Tracks
            # Using VGGSfM tracker instead of VGGT tracker for efficiency
            # VGGT tracker requires multiple backbone runs to query different frames (this is a problem caused by the training process)
            # Will be fixed in VGGT v2

            # You can also change the pred_tracks to tracks from any other methods
            # e.g., from COLMAP, from CoTracker, or by chaining 2D matches from Lightglue/LoFTR.
            pred_tracks, pred_vis_scores, pred_confs, points_3d, points_rgb = predict_tracks(
                images,
                conf=depth_conf,
                points_3d=points_3d,
                masks=None,
                max_query_pts=args.max_query_pts,
                query_frame_num=args.query_frame_num,
                keypoint_extractor="aliked+sp",
                fine_tracking=args.fine_tracking,
            )

            torch.cuda.empty_cache()

        # rescale the intrinsic matrix from 518 to 1024
        intrinsic[:, :2, :] *= scale
        track_mask = pred_vis_scores > args.vis_thresh

        # TODO: radial distortion, iterative BA, masks
        reconstruction, valid_track_mask = batch_np_matrix_to_pycolmap(
            points_3d,
            extrinsic,
            intrinsic,
            pred_tracks,
            image_size,
            masks=track_mask,
            max_reproj_error=args.max_reproj_error,
            shared_camera=shared_camera,
            camera_type=args.camera_type,
            points_rgb=points_rgb,
        )

        if reconstruction is None:
            raise ValueError("No reconstruction can be built with BA")

        # Bundle Adjustment
        ba_options = pycolmap.BundleAdjustmentOptions()
        pycolmap.bundle_adjustment(reconstruction, ba_options)

        reconstruction_resolution = img_load_resolution
    else:
        conf_thres_value = args.conf_thres_value
        # conf_thres_value = np.percentile(depth_conf, 70)
        max_points_for_colmap = args.max_points  # randomly sample 3D points
        shared_camera = False  # in the feedforward manner, we do not support shared camera
        camera_type = "PINHOLE"  # in the feedforward manner, we only support PINHOLE camera

        image_size = np.array([vggt_fixed_resolution, vggt_fixed_resolution])
        num_frames, height, width, _ = points_3d.shape
        images = images.float()

        points_rgb = F.interpolate(
            images, size=(vggt_fixed_resolution, vggt_fixed_resolution), mode="bilinear", align_corners=False
        )
        points_rgb = (points_rgb.cpu().numpy() * 255).astype(np.uint8)
        points_rgb = points_rgb.transpose(0, 2, 3, 1)

        # (S, H, W, 3), with x, y coordinates and frame indices
        points_xyf = create_pixel_coordinate_grid(num_frames, height, width)

        conf_mask = depth_conf >= conf_thres_value
        # at most writing 100000 3d points to colmap reconstruction object
        conf_mask = randomly_limit_trues(conf_mask, max_points_for_colmap)

        points_3d = points_3d[conf_mask]
        points_xyf = points_xyf[conf_mask]
        points_rgb = points_rgb[conf_mask]

        print("Converting to COLMAP format")
        reconstruction = batch_np_matrix_to_pycolmap_wo_track(
            points_3d,
            points_xyf,
            points_rgb,
            extrinsic,
            intrinsic,
            image_size,
            shared_camera=shared_camera,
            camera_type=camera_type,
        )

        reconstruction_resolution = vggt_fixed_resolution

    reconstruction = rename_colmap_recons_and_rescale_camera(
        reconstruction,
        base_image_path_list,
        original_coords.cpu().numpy(),
        img_size=reconstruction_resolution,
        shift_point2d_to_original_res=True,
        shared_camera=shared_camera,
    )

    print(f"Saving reconstruction to {args.output_dir}/sparse/0")
    sparse_reconstruction_dir = os.path.join(args.output_dir, "sparse/0")
    os.makedirs(sparse_reconstruction_dir, exist_ok=True)
    reconstruction.write(sparse_reconstruction_dir)

    # Save point cloud for fast visualization
    trimesh.PointCloud(points_3d, colors=points_rgb).export(os.path.join(args.output_dir, "sparse/0/points.ply"))


    # print time recorder results
    wall_s, cuda_s = tm.span("before_load_images", "after_load_images")
    print(f"load vggt: "f"wall={wall_s:.3f}s | "f"cuda={cuda_s:.3f}s")
    wall_s, cuda_s = tm.span("before_run_vggt", "after_run_vggt")
    print(f"run vggt: "f"wall={wall_s:.3f}s | "f"cuda={cuda_s:.3f}s")

    if args.test_sparse_view:
        wall_s, cuda_s = tm.span("before_find_query_poses", "after_find_query_poses")
        print(f"find query poses: "f"wall={wall_s:.3f}s | "f"cuda={cuda_s:.3f}s")

    return True


def rename_colmap_recons_and_rescale_camera(
    reconstruction, image_paths, original_coords, img_size, shift_point2d_to_original_res=False, shared_camera=False
):
    rescale_camera = True

    for pyimageid in reconstruction.images:
        # Reshaped the padded&resized image to the original size
        # Rename the images to the original names
        pyimage = reconstruction.images[pyimageid]
        pycamera = reconstruction.cameras[pyimage.camera_id]
        pyimage.name = image_paths[pyimageid - 1]

        if rescale_camera:
            # Rescale the camera parameters
            pred_params = copy.deepcopy(pycamera.params)

            real_image_size = original_coords[pyimageid - 1, -2:]
            resize_ratio = max(real_image_size) / img_size
            pred_params = pred_params * resize_ratio
            real_pp = real_image_size / 2
            pred_params[-2:] = real_pp  # center of the image

            pycamera.params = pred_params
            pycamera.width = real_image_size[0]
            pycamera.height = real_image_size[1]

        if shift_point2d_to_original_res:
            # Also shift the point2D to original resolution
            top_left = original_coords[pyimageid - 1, :2]

            for point2D in pyimage.points2D:
                point2D.xy = (point2D.xy - top_left) * resize_ratio

        if shared_camera:
            # If shared_camera, all images share the same camera
            # no need to rescale any more
            rescale_camera = False

    return reconstruction



if __name__ == "__main__":
    args = parse_args()
    
    with torch.no_grad():
        demo_fn(args)
    # if args.adjust_folder:
    #     restructure_scene_dir(args)


# Work in Progress (WIP)

"""
VGGT Runner Script
=================

A script to run the VGGT model for 3D reconstruction from image sequences.

Directory Structure
------------------
Input:
    input_folder/
    └── images/            # Source images for reconstruction

Output:
    output_folder/
    ├── images/
    ├── sparse/           # Reconstruction results
    │   ├── cameras.bin   # Camera parameters (COLMAP format)
    │   ├── images.bin    # Pose for each image (COLMAP format)
    │   ├── points3D.bin  # 3D points (COLMAP format)
    │   └── points.ply    # Point cloud visualization file 
    └── visuals/          # Visualization outputs TODO

Key Features
-----------
• Dual-mode Support: Run reconstructions using either VGGT or VGGT+BA
• Resolution Preservation: Maintains original image resolution in camera parameters and tracks
• COLMAP Compatibility: Exports results in standard COLMAP sparse reconstruction format
"""
