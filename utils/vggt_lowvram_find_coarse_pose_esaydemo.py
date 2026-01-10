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

# for deterministic behavior
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)

import json
import math

from factory.vggt_low_vram.vggt.models.vggt import VGGT
from factory.vggt_low_vram.vggt.utils.load_fn import load_and_preprocess_images_square
from factory.vggt_low_vram.vggt.utils.pose_enc import pose_encoding_to_extri_intri

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


def find_coarse_pose(seed, scene_dir, transforms_out, output_dir, image_base_dir):
    # Set seed for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU
    print(f"Setting seed as: {seed}")

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
    print(f"Model loaded")

    # Get image paths and preprocess them
    image_dir = os.path.join(scene_dir, "images")
    image_path_list = glob.glob(os.path.join(image_dir, "*"))
    if len(image_path_list) == 0:
        raise ValueError(f"No images found in {image_dir}")
    base_image_path_list = [os.path.basename(path) for path in image_path_list]

    # Load images and original coordinates
    # Load Image in 1024, while running VGGT with 518
    vggt_fixed_resolution = 518
    img_load_resolution = 1024

    images, original_coords = load_and_preprocess_images_square(image_path_list, img_load_resolution)
    print(f"Loaded {len(images)} images from {image_dir}")

    # Run VGGT to estimate camera and depth
    # Run with 518x518 images
    extrinsic, intrinsic, depth_map, depth_conf = run_VGGT(model, images, device, dtype, vggt_fixed_resolution)

    out_path = os.path.join(output_dir, transforms_out)

    write_transforms_json_from_vggt(
        extrinsic_w2c=extrinsic,
        intrinsic=intrinsic,
        image_names=base_image_path_list,
        original_coords=original_coords.cpu().numpy() if torch.is_tensor(original_coords) else original_coords,
        img_size=vggt_fixed_resolution,  # because extrinsic/intrinsic come from 518
        out_path=out_path,
        image_base_dir=image_base_dir,
    )

    # If you only want transforms.json, we can stop here.
    return True
        
def opencv_to_opengl(T_c2w: np.ndarray) -> np.ndarray:
    fix = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float64)
    return T_c2w @ fix


def write_transforms_json_from_vggt(
    extrinsic_w2c: np.ndarray,   # (N,4,4) or (N,3,4)
    intrinsic: np.ndarray,       # (N,3,3) or (3,3)
    image_names: list,           # length N, e.g. base_image_path_list
    original_coords: np.ndarray, # (N,4) typically [top_left_x, top_left_y, W, H]
    img_size: int,               # vggt_fixed_resolution or img_load_resolution used for VGGT intrinsics
    out_path: str,
    image_base_dir: str | None = None,
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

    if len(image_names) != N:
        raise ValueError(f"len(image_names)={len(image_names)} != N={N}")

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
        # set principal point to image center (same as your rename_colmap... logic)
        K[0, 2] = real_wh[0] / 2.0
        K[1, 2] = real_wh[1] / 2.0

        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])

        file_path = image_names[i]
        if image_base_dir is not None:
            file_path = os.path.join(image_base_dir, file_path).replace("\\", "/")
        else:
            file_path = file_path.replace("\\", "/")

        frames.append({
            "file_path": file_path,
            "transform_matrix": c2w.tolist(),
            "fl_x": fx, "fl_y": fy,
            "cx": cx, "cy": cy,
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


if __name__ == "__main__":
    seed = 42
    scene_dir="/home/wangyifa/tmp/3dfm4anomaly_detection/scripts/test_MAD_Sim_vggt_find_queryimg_coarse_pose/3dgs_model/01Gorilla_with_queryimg_testing"
    transforms_out = "transforms_testing.json"
    output_dir = "/home/wangyifa/tmp/3dfm4anomaly_detection/scripts/test_MAD_Sim_vggt_find_queryimg_coarse_pose/3dgs_model/01Gorilla_with_queryimg_testing"
    image_base_dir="/home/wangyifa/tmp/3dfm4anomaly_detection/scripts/test_MAD_Sim_vggt_find_queryimg_coarse_pose/3dgs_model/01Gorilla_with_queryimg_testing/images"

    with torch.no_grad():
        find_coarse_pose(seed, scene_dir, transforms_out, output_dir, None)