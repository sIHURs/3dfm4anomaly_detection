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
import tqdm
from datetime import datetime
import math
now = datetime.now()

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
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
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
    parser.add_argument("--save_jsonl", action="store_true", help="also append query poses to a merged jsonl file")
    parser.add_argument("--eval_dir", help="dir that contains burrs/good/missing/stains")
    parser.add_argument("--test_sparse_view", action="store_true", default=False, help="test with sparse view input")

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
    print(f"[OK][{now}] wrote transforms.json: {out_path}  (#frames={N})")


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
    model = VGGT()
    _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
    model.eval()
    model = model.to(dtype=dtype, device=device)
    print(f"-- Model loaded --")

    # Get image paths and preprocess them
    image_dir = os.path.join(args.scene_dir, "images")
    image_path_list = list_images_sorted(image_dir)
    if len(image_path_list) == 0:
        raise ValueError(f"No images found in {image_dir}")

    print(f"[OK][{now}] train images: {len(image_path_list)} from {image_dir}")

    base_image_path_list = [os.path.basename(path) for path in image_path_list]

    # Load images and original coordinates
    # Load Image in 1024, while running VGGT with 518
    vggt_fixed_resolution = 518
    img_load_resolution = 1024

    images, original_coords = load_and_preprocess_images_square(image_path_list, img_load_resolution)
    print(f"[OK][{now}] Loaded {len(images)} images from {image_dir}")

    # Run VGGT to estimate camera and depth
    # Run with 518x518 images
    extrinsic, intrinsic, depth_map, depth_conf = run_VGGT(model, images, device, dtype, vggt_fixed_resolution)
    points_3d = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)
    # images = images.float()

    # save as blender - transforms.json
    print(f"[OK][{now}] train poses computed: {extrinsic.shape[0]} images")
    out_name = f"transforms_anomaly_free_poses.json"
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
            raise RuntimeError(f"No query images found under {args.eval_dir}/{{burrs,good,missing,stains}}")
        print(f"[OK][{now}] total queries: {len(all_queries)}")

        jsonl_path = os.path.join(args.out_dir, "query_poses_merged.jsonl")
        if args.save_jsonl and os.path.exists(jsonl_path):
            os.remove(jsonl_path)

        for idx, (subset, qpath) in enumerate(
            tqdm(all_queries, desc="Processing queries", unit="img")
        ):
            # load single query
            q_imgs, q_coords_t = load_and_preprocess_images_square([qpath], args.img_load_resolution)
            q_coords = q_coords_t.cpu().numpy() if torch.is_tensor(q_coords_t) else q_coords_t

            # pack = train + query (no file copy)
            packed_imgs = torch.cat([images, q_imgs], dim=0)
            packed_coords = np.concatenate([original_coords, q_coords], axis=0)
            packed_paths = image_path_list + [qpath]

            packed_paths_name = [os.path.basename(p) for p in packed_paths]

            # run vggt
            extri, intri, _, _ = run_VGGT(model, packed_imgs, device, dtype, args.vggt_resolution)

            # write per-query transforms (train + this query)
            out_name = f"transforms_{subset}_{idx:05d}_{safe_stem(qpath)}.json"
            out_path = os.path.join(args.out_dir, "verbose_transforms_file", out_name)
            write_transforms_json_from_vggt(
                extrinsic_w2c=extri,
                intrinsic=intri,
                image_paths=packed_paths_name,
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

        print("[DONE][{now}] all queries processed")


    if args.save_depth:
        os.makedirs(args.output_dir, exist_ok=True)
        depth_map_dir = os.path.join(args.output_dir, "verbose", "depth_map")
        depth_conf_dir = os.path.join(args.output_dir, "verbose", "depth_conf_map")
        os.makedirs(depth_map_dir, exist_ok=True)
        os.makedirs(depth_conf_dir, exist_ok=True)
        
        save_depth_outputs(depth_map, depth_conf, out_dir=args.output_dir, prefix="vggt")

        for i in range(depth_map.shape[0]):
            # save depth map
            save_depth_png(
                depth_map[i],
                os.path.join(args.output_dir, "verbose", "depth_map", f"depth_{i:03d}.png")
            )
            # save depth confidence map
            c = depth_conf[i]
            c_np = c.detach().float().cpu().numpy() if torch.is_tensor(c) else c.astype(np.float32)
            vmin = np.percentile(c_np, 5)
            vmax = np.percentile(c_np, 95)
            save_depth_png(
                c_np,
                os.path.join(args.output_dir, "verbose", "depth_conf_map", f"depth_conf_{i:03d}.png"),
                vmin=vmin,
                vmax=vmax
            )

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

    print(f"Saving reconstruction to {args.output_dir}/sparse")
    sparse_reconstruction_dir = os.path.join(args.output_dir, "sparse")
    os.makedirs(sparse_reconstruction_dir, exist_ok=True)
    reconstruction.write(sparse_reconstruction_dir)

    # Save point cloud for fast visualization
    trimesh.PointCloud(points_3d, colors=points_rgb).export(os.path.join(args.output_dir, "sparse/points.ply"))

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
    if args.adjust_folder:
        restructure_scene_dir(args)


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
