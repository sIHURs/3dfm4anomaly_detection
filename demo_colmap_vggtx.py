# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random
import numpy as np
import glob
import os
import torch
import torch.nn.functional as F
import shutil
import math
import json
import cv2

# disable triton if arch not support
def is_pascal():
    if not torch.cuda.is_available():
        print("❌ No CUDA device available.")
        return False
    major, minor = torch.cuda.get_device_capability()
    print(f"🔍 CUDA Capability: {major}.{minor}")
    return major == 6  # Pascal = 6.x

if is_pascal():
    print("✅ Detected Pascal GPU — disabling advanced features...")
    os.environ["TORCHINDUCTOR_DISABLE"] = "1"
    os.environ["TORCH_COMPILE_DISABLE"] = "1"
    os.environ["TORCHDYNAMO_DISABLE"] = "1"
    os.environ["TRITON_DISABLE"] = "1"
    os.environ["TORCH_CUDA_FUSER_DISABLE"] = "1"
else:
    print("🚀 Non-Pascal GPU — using optimized TorchInductor if available.")


import torch._dynamo
torch._dynamo.config.suppress_errors = True
# disable triton check
try:
    torch._inductor
    print("⚙️ TorchInductor on")
except AttributeError:
    print("✅ TorchInductor off")

print("Device capability:", torch.cuda.get_device_capability())

# Configure CUDA settings
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

import argparse
import trimesh
import factory.vggtx.utils.opt as opt_utils
import utils.colmap as colmap_utils
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
now = datetime.now()
from utils.time_recorder import SpanTimer
from factory.vggtx.utils.metric_torch import evaluate_auc, evaluate_pcd, write_evaluation_results

from factory.vggtx.vggt.models.vggt import VGGT
from factory.vggtx.vggt.utils.load_fn import load_and_preprocess_images_ratio
from factory.vggtx.vggt.utils.pose_enc import pose_encoding_to_extri_intri
from factory.vggtx.vggt.utils.geometry import unproject_depth_map_to_point_map
from factory.vggtx.vggt.utils.helper import create_pixel_coordinate_grid, randomly_limit_trues
from factory.vggtx.vggt.dependency.np_to_pycolmap import batch_np_matrix_to_pycolmap_wo_track


torch._dynamo.config.accumulated_cache_size_limit = 512

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

    print(f"[OK][{now}] Saved depth_map and depth_conf to {out_dir}")

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

def restructure_scene_dir(scene_dir: str):
    print(f"[INFO] Restructuring scene dir: {scene_dir}")

    # 1) images -> input
    src_images = os.path.join(scene_dir, "images")
    dst_input  = os.path.join(scene_dir, "input")

    if os.path.exists(src_images):
        if os.path.exists(dst_input):
            print(f"[SKIP] '{dst_input}' already exists, keep it.")
        else:
            shutil.move(src_images, dst_input)
            print(f"[OK] Moved 'images' -> 'input'")
    else:
        print(f"[SKIP] No 'images' directory found.")

    # 2) sparse/0 -> distorted/sparse/0
    src_sparse0 = os.path.join(scene_dir, "sparse", "0")
    dst_sparse_parent = os.path.join(scene_dir, "distorted", "sparse")
    dst_sparse0 = os.path.join(dst_sparse_parent, "0")

    if os.path.exists(src_sparse0):
        if os.path.exists(dst_sparse0):
            print(f"[SKIP] '{dst_sparse0}' already exists, keep it.")
        else:
            os.makedirs(dst_sparse_parent, exist_ok=True)
            shutil.move(src_sparse0, dst_sparse_parent)  # move folder "0"
            print(f"[OK] Moved 'sparse/0' -> 'distorted/sparse/0'")
    else:
        print(f"[SKIP] No 'sparse/0' directory found.")

    # 3) remove old sparse if empty
    src_sparse_root = os.path.join(scene_dir, "sparse")
    if os.path.exists(src_sparse_root):
        remaining = os.listdir(src_sparse_root)
        if len(remaining) == 0:
            shutil.rmtree(src_sparse_root)
            print(f"[OK] Removed empty 'sparse' directory")
        else:
            print(f"[WARN] 'sparse' not empty, keeping it: {remaining}")
    else:
        print(f"[SKIP] No 'sparse' directory found.")

    print(f"[DONE] Folder structure updated.\n")
    

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



def run_VGGT(images, device, dtype, chunk_size, tm=None):
    # images: [B, 3, H, W]

    # Run VGGT for camera and depth estimation
    tm.mark("before_load_images")
    model = VGGT(chunk_size=chunk_size)
    _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
    model.eval()
    model = model.to(device).to(dtype)
    tm.mark("after_load_images")
    model.track_head = None  # we do not need tracking head for reconstruction
    print(f"Model loaded")

    with torch.no_grad():
        tm.mark("before_run_vggt")
        predictions = model(images.to(device, dtype), verbose=True)
        tm.mark("after_run_vggt")
        extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions['pose_enc'], images.shape[-2:])
        extrinsic = extrinsic.squeeze(0).cpu().numpy()
        intrinsic = intrinsic.squeeze(0).cpu().numpy()
        depth_map = predictions['depth'].squeeze(0).cpu().numpy()
        depth_conf = predictions['depth_conf'].squeeze(0).cpu().numpy()
    
    return extrinsic, intrinsic, depth_map, depth_conf

def parse_args():
    parser = argparse.ArgumentParser(description="VGGT Demo")
    parser.add_argument("--scene_dir", type=str, default="data/MAD_Scene", help="Directory containing the scene images")
    parser.add_argument("--post_fix", type=str, default="_vggt_x", help="Post fix for the output folder")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument("--use_ga", action="store_true", default=False, help="Whether to apply global alignment for better reconstruction")
    parser.add_argument("--save_depth", action="store_true", default=False, help="If save depth")
    parser.add_argument("--chunk_size", type=int, default=256, help="Chunk size for frame-wise operation in VGGT")
    parser.add_argument("--total_frame_num", type=int, default=None, help="Number of frames to reconstruct")
    ######### GA parameters #########
    parser.add_argument("--max_query_pts", type=int, default=None, help="Maximum number of query points")
    parser.add_argument("--max_points_for_colmap", type=int, default=100000, help="Maximum number for colmap point cloud") # the default from vggtx is 500000
    parser.add_argument("--shared_camera", action="store_true", default=False, help="Use shared camera for all images")
    parser.add_argument(
        "--conf_thres_value", type=float, default=5.0, help="Confidence threshold value for depth filtering (wo BA)"
    )
    
    # todo: for sparse view experiments
    parser.add_argument("--eval_dir", help="dir that contains burrs/good/missing/stains")
    parser.add_argument("--test_sparse_view", action="store_true", default=False, help="test with sparse view input")
    parser.add_argument("--query_batch_size", type=int, default=1, help="test with sparse view input")
    return parser.parse_args()

def demo_fn(args):
    # Print configuration
    print("Arguments:", vars(args))

    # target_scene_dir = os.path.join(f"{os.path.dirname(args.scene_dir)}{args.post_fix}", os.path.basename(args.scene_dir))
    # os.makedirs(target_scene_dir, exist_ok=True)
    target_scene_dir = os.path.join(args.scene_dir)

    # # set time recorder
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

    # Get image paths and preprocess them
    image_dir = os.path.join(args.scene_dir, "images")
    if args.total_frame_num is None:
        args.total_frame_num = len(os.listdir(image_dir))

    if os.path.exists(os.path.join(args.scene_dir, "sparse/0/images.bin")):
        print("Using order of ground truth images from COLMAP sparse reconstruction")
        images_gt = colmap_utils.read_images_binary(os.path.join(args.scene_dir, "sparse/0/images.bin"))
        assert args.total_frame_num <= len(images_gt), f"Requested total_frame_num {args.total_frame_num} exceeds available images {len(images_gt)}"
        
        images_gt = dict(list(images_gt.items())[:args.total_frame_num])
        images_gt_keys = list(images_gt.keys())

        random.shuffle(images_gt_keys)
        images_gt_updated = {id: images_gt[id] for id in list(images_gt_keys)}
        image_path_list = [os.path.join(image_dir, images_gt_updated[id].name) for id in images_gt_updated.keys()]

        inverse_idx = [images_gt_keys.index(key) for key in sorted(list(images_gt.keys()))]
    else:
        image_path_list = sorted(glob.glob(os.path.join(image_dir, "*")))[:args.total_frame_num]
        if not image_path_list:
            raise ValueError(f"No images found in {image_dir}")
        inverse_idx = list(range(len(image_path_list)))

    base_image_path_list = [os.path.basename(path) for path in image_path_list]
    base_image_path_list_inv = [base_image_path_list[i] for i in inverse_idx]

    # Load images and original coordinates
    # Load Image in 1024, while running VGGT with 518
    img_load_resolution = 518

    images, original_coords = load_and_preprocess_images_ratio(image_path_list, img_load_resolution)
    original_coords = original_coords.to(device)
    print(f"Loaded {len(images)} images from {image_dir}")

    torch.cuda.reset_peak_memory_stats()

    # Run VGGT to estimate camera and depth
    # Run with 518x518 images
    extrinsic, intrinsic, depth_map, depth_conf = run_VGGT(images, device, dtype, args.chunk_size, tm)
    
    images = images.to(device)
    
    if args.use_ga:
        tm.mark("before_ga")
        if os.path.exists(os.path.join(target_scene_dir, "matches.pt")):
            print(f"Found existing matches at {os.path.join(target_scene_dir, 'matches.pt')}, loading it")
            match_outputs = torch.load(
                os.path.join(target_scene_dir, "matches.pt"),
                map_location="cpu",
                weights_only=False,
            )
        else:
            print("Extracting matches for global alignment")
            if args.max_query_pts is None:
                args.max_query_pts = 4096 if len(images) < 500 else 2048
            match_outputs = opt_utils.extract_matches(extrinsic, intrinsic, images, depth_conf, base_image_path_list, args.max_query_pts)
            match_outputs["original_width"] = images.shape[-1]
            match_outputs["original_height"] = images.shape[-2]
            torch.save(match_outputs, os.path.join(target_scene_dir, "matches.pt"))
            print(f"Saved matches to {os.path.join(target_scene_dir, 'matches.pt')}")
        extrinsic, intrinsic = opt_utils.pose_optimization(
            match_outputs, extrinsic, intrinsic, images, depth_map, depth_conf,
            base_image_path_list, target_scene_dir=target_scene_dir, shared_intrinsics=args.shared_camera,
        )
        tm.mark("after_ga")

    images = images.detach().cpu()
    # conf_thres_value = np.percentile(depth_conf, 0.5)
    conf_thres_value = args.conf_thres_value
    print(f"Using confidence threshold: {conf_thres_value}")
    shared_camera = False  # in colmap result saving, we do not support shared camera
    camera_type = "PINHOLE"  # in colmap result saving, we only support PINHOLE camera

    # c = 2.5  # scale factor for better reconstruction, hard-coded here
    # extrinsic[:, :3, 3] *= c
    # depth_map *= c

    points_3d = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)

    print(f"[OK][{now}] train poses computed: {extrinsic.shape[0]} images")
    out_name = f"transforms_anomaly_free_poses_uncentered.json"
    out_path = os.path.join(target_scene_dir, out_name)
    write_transforms_json_from_vggt(
        extrinsic_w2c=extrinsic,
        intrinsic=intrinsic,
        image_paths=base_image_path_list,
        original_coords=original_coords.cpu().numpy() if torch.is_tensor(original_coords) else original_coords,
        img_size=img_load_resolution,
        out_path=out_path,
    )

    if args.test_sparse_view:
        tm.mark("before_find_query_poses")
        print(f"TESTING SPARSE VIEW INPUT")
        print(f"[OK][{now}] Preparing query images from {args.eval_dir}")

        subsets = ["Burrs", "good", "Missing", "Stains"]
        all_queries = []
        for s in subsets:
            d = os.path.join(args.eval_dir, s)
            if os.path.isdir(d):
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
        q_imgs, q_coords_t = load_and_preprocess_images_ratio(qpaths_all, img_load_resolution)
        q_coords = q_coords_t.cpu().numpy() if torch.is_tensor(q_coords_t) else q_coords_t

        # ------------------------------------------------------------
        # 3) Pack: 10 train + ALL queries, run VGGT ONCE
        # ------------------------------------------------------------
        packed_imgs = torch.cat([images_10, q_imgs], dim=0)
        original_coords_10 = original_coords_10.detach().cpu().numpy() if torch.is_tensor(original_coords_10) else original_coords_10
        packed_coords = np.concatenate([original_coords_10, q_coords], axis=0)
        packed_paths = image_path_list_10 + qpaths_all
        packed_paths_name = [os.path.basename(p) for p in packed_paths]

        print(f"[OK][{now}] Packed {len(image_path_list_10)} train + {len(qpaths_all)} query = {len(packed_paths)} total frames")

        extri, intri, depth_map_query, depth_conf_query = run_VGGT(packed_imgs, device, dtype, img_load_resolution, tm)

        out_path = os.path.join(target_scene_dir, "transforms_query_poses_uncentered.json")
        write_transforms_json_from_vggt(
            extrinsic_w2c=extri,
            intrinsic=intri,
            image_paths=packed_paths_name,
            original_coords=packed_coords,
            img_size=img_load_resolution,
            out_path=out_path,
        )

        print(f"[DONE][{now}] all queries processed")
        tm.mark("after_find_query_poses")

        if args.save_depth:
            os.makedirs(target_scene_dir, exist_ok=True)
            depth_map_dir = os.path.join(target_scene_dir, "verbose", "depth_map_query")
            depth_conf_dir = os.path.join(target_scene_dir, "verbose", "depth_conf_map_query")
            os.makedirs(depth_map_dir, exist_ok=True)
            os.makedirs(depth_conf_dir, exist_ok=True)
            
            save_depth_outputs(depth_map_query, depth_conf_query, out_dir=target_scene_dir, prefix="vggt_query")

            for i in range(depth_map_query.shape[0]):
                import matplotlib.pyplot as plt

                base_name = os.path.splitext(packed_paths_name[i])[0]

                save_depth_png(
                    depth_map_query[i],
                    os.path.join(
                        target_scene_dir,
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
                    target_scene_dir,
                    "verbose",
                    "depth_conf_map_query",
                    f"depth_conf_query_heatmap_{base_name}.png",
                )
                cv2.imwrite(out_path, conf_heatmap)

    if args.save_depth:
        os.makedirs(target_scene_dir, exist_ok=True)
        depth_map_dir = os.path.join(target_scene_dir, "verbose", "depth_map")
        depth_conf_dir = os.path.join(target_scene_dir, "verbose", "depth_conf_map")
        conf_points_dir = os.path.join(target_scene_dir, "verbose", "conf_points")
        conf_hist_dir = os.path.join(target_scene_dir, "verbose", "depth_conf_hist")
        os.makedirs(depth_map_dir, exist_ok=True)
        os.makedirs(depth_conf_dir, exist_ok=True)
        os.makedirs(conf_points_dir, exist_ok=True)
        os.makedirs(conf_hist_dir, exist_ok=True)
        
        save_depth_outputs(depth_map, depth_conf, out_dir=target_scene_dir, prefix="vggt")

        for i in range(depth_map.shape[0]):
            import matplotlib.pyplot as plt

            base_name = os.path.splitext(base_image_path_list[i])[0]

            save_depth_png(
                depth_map[i],
                os.path.join(
                    target_scene_dir,
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
                target_scene_dir,
                "verbose",
                "depth_conf_map",
                f"depth_conf_heatmap_{base_name}.png",
            )
            cv2.imwrite(out_path, conf_heatmap)

            ## if normalize
            # c_proc = np.clip(c_np, vmin, vmax)
            # c_proc = (c_proc - vmin) / (vmax - vmin + 1e-6)

            ## if mask
            # depth_i = depth_map[i].detach().float().cpu().numpy() if torch.is_tensor(depth_map[i]) else depth_map[i]
            # mask = np.isfinite(depth_i) & (depth_i > 0)
            # vals = c_proc[mask].reshape(-1)
            vals = c_np.reshape(-1)

            fig = plt.figure()
            plt.hist(vals, bins=50)
            plt.title(f"depth_conf histogram: {base_name}\nclip p5={vmin:.3g}, p95={vmax:.3g}")
            plt.xlabel("conf (clipped & normalized to [0,1])")
            plt.ylabel("count")
            hist_path = os.path.join(conf_hist_dir, f"depth_conf_hist_{base_name}.png")
            plt.tight_layout()
            plt.savefig(hist_path, dpi=150)
            plt.close(fig)

            pts_depth = topk_points_nms(c_np, k=10, min_dist=15, shape="diamond")
            depth_size = int(c_np.shape[-1])         # e.g. 518
            square_size = int(images.shape[-1])      # e.g. 1024

            pts_orig = map_points_depth_to_original(
                pts_depth,
                original_coords[i].detach().cpu().numpy(),
                depth_size=depth_size,
                square_size=square_size,
            )

            orig_path = image_path_list[i]
            orig_bgr = cv2.imread(orig_path, cv2.IMREAD_COLOR)
            if orig_bgr is None:
                raise ValueError(f"Failed to read image: {orig_path}")
            
            conf_vis = draw_points_on_image(orig_bgr, pts_orig)

            out_path = os.path.join(
                target_scene_dir,
                "verbose",
                "conf_points",
                f"depth_conf_top3_on_orig_{base_name}.png",
            )

            ok = cv2.imwrite(out_path, conf_vis)
            if not ok:
                raise IOError(
                    f"cv2.imwrite failed: {out_path}, "
                    f"shape={conf_vis.shape}, dtype={conf_vis.dtype}"
                )

    image_size = np.array([depth_map.shape[1], depth_map.shape[2]])
    num_frames, height, width, _ = points_3d.shape

    points_rgb = F.interpolate(
        images, size=(depth_map.shape[1], depth_map.shape[2]), mode="bilinear", align_corners=False
    )
    points_rgb = (points_rgb.cpu().numpy() * 255).astype(np.uint8)
    points_rgb = points_rgb.transpose(0, 2, 3, 1)

    # (S, H, W, 3), with x, y coordinates and frame indices
    points_xyf = create_pixel_coordinate_grid(num_frames, height, width)

    if args.use_ga:
        conf_mask = opt_utils.extract_conf_mask(match_outputs, depth_conf, base_image_path_list)
        conf_mask = conf_mask & (depth_conf >= conf_thres_value)
        conf_mask = randomly_limit_trues(conf_mask, args.max_points_for_colmap)
    else:
        conf_mask = depth_conf >= conf_thres_value
        # at most writing args.max_points_for_colmap 3d points to colmap reconstruction object
        conf_mask = randomly_limit_trues(conf_mask, args.max_points_for_colmap)

    points_3d = points_3d[conf_mask]
    points_xyf = points_xyf[conf_mask]
    points_rgb = points_rgb[conf_mask]

    print("Converting to COLMAP format")
    reconstruction = batch_np_matrix_to_pycolmap_wo_track(
        points_3d,
        points_xyf,
        points_rgb,
        extrinsic[inverse_idx],
        intrinsic[inverse_idx],
        image_size,
        shared_camera=shared_camera,
        camera_type=camera_type,
    )

    reconstruction_resolution = (depth_map.shape[2], depth_map.shape[1])

    reconstruction = colmap_utils.rename_colmap_recons_and_rescale_camera(
        reconstruction,
        base_image_path_list_inv,
        original_coords.cpu().numpy()[inverse_idx],
        img_size=reconstruction_resolution,
        shift_point2d_to_original_res=True,
        shared_camera=shared_camera,
    )

    # first create a folder named f"{args.scene_dir}_vggt", then soft link everything from args.scene_dir except for "sparse"
    for item in os.listdir(args.scene_dir):
        if item != "sparse" and not item.endswith("results.txt"):
            src = os.path.join(args.scene_dir, item)
            dst = os.path.join(target_scene_dir, item)
            if os.path.isdir(src):
                os.makedirs(dst, exist_ok=True)
                for file in os.listdir(src):
                    if not os.path.exists(os.path.join(dst, file)):
                        os.symlink(os.path.abspath(os.path.join(src, file)), os.path.abspath(os.path.join(dst, file)))
            else:
                if not os.path.exists(dst):
                    os.symlink(os.path.abspath(src), os.path.abspath(dst))

    print(f"Saving reconstruction to {target_scene_dir}/sparse/0")
    sparse_reconstruction_dir = os.path.join(target_scene_dir, "sparse/0")
    os.makedirs(sparse_reconstruction_dir, exist_ok=True)
    reconstruction.write(sparse_reconstruction_dir)

    # Save point cloud for fast visualization
    trimesh.PointCloud(points_3d, colors=points_rgb).export(os.path.join(target_scene_dir, "sparse/0/points.ply"))

    # restructure_scene_dir(target_scene_dir)

    # print time recorder results
    wall_s, cuda_s = tm.span("before_load_images", "after_load_images")
    print(f"[Duration Recorder] load vggt: "f"wall={wall_s:.3f}s | "f"cuda={cuda_s:.3f}s")
    wall_s, cuda_s = tm.span("before_run_vggt", "after_run_vggt")
    print(f"[Duration Recorder] run vggt: "f"wall={wall_s:.3f}s | "f"cuda={cuda_s:.3f}s")
    if args.test_sparse_view:
        wall_s, cuda_s = tm.span("before_find_query_poses", "after_find_query_poses")
        print(f"find query poses: "f"wall={wall_s:.3f}s | "f"cuda={cuda_s:.3f}s")
    wall_s, cuda_s = tm.span("before_ga", "after_ga")
    print(f"[Duration Recorder] ga duration: "f"wall={wall_s:.3f}s | "f"cuda={cuda_s:.3f}s")

    return True



if __name__ == "__main__":
    args = parse_args()
    demo_fn(args)


# Work in Progress (WIP)

"""
VGGT-X Runner Script
=================

A script to run the VGGT-X model for 3D reconstruction from image sequences.

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
• Dual-mode Support: Run reconstructions using either VGGT or VGGT+GA
• Resolution Preservation: Maintains original image resolution in camera parameters and tracks
• COLMAP Compatibility: Exports results in standard COLMAP sparse reconstruction format
"""
