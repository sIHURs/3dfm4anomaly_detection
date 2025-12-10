import os
import argparse

import cv2
import numpy as np
import torch
from plyfile import PlyData, PlyElement
from tqdm import tqdm

import colmap


def load_colmap_model(colmap_dir: str):
    """
    Load COLMAP cameras, images and points3D from a sparse model directory.
    """
    cameras, images, points3D = colmap.read_model(colmap_dir, ext=".bin")
    return cameras, images, points3D


def detect_ply_type(vertex, fallback: str = "pointcloud") -> str:
    """
    Detect whether a PLY file is a 3D Gaussian Splatting PLY (3DGS)
    or a normal point cloud based on vertex property names.
    """
    names = set(vertex.data.dtype.names or [])

    # Typical 3DGS attributes (gaussian-splatting, splatfacto, etc.)
    three_dgs_signals = {
        "scale_0", "scale_1", "scale_2",
        "rotation",
        "opacity",
        "f_dc_0", "f_rest_0", "f_rest_1",
        "sh0", "sh1", "sh2",
    }

    if names & three_dgs_signals:
        return "3dgs"

    # Otherwise treat as normal point cloud
    return fallback


@torch.no_grad()
def filter_gaussians_torch(
    pts_np: np.ndarray,
    cameras,
    images,
    mask_cache_np: dict,
    outside_threshold: float = 0.6,
    device: str = "cuda",
) -> np.ndarray:
    """
    Torch/CUDA implementation of the alpha-mask filtering.

    Args:
        pts_np:          (N, 3) numpy array of xyz coordinates.
        cameras:         COLMAP cameras dict.
        images:          COLMAP images dict.
        mask_cache_np:   dict[image_id] -> np.ndarray mask (H,W), uint8 (0 or 255).
        outside_threshold: points with outside_ratio >= this will be removed.
        device:          "cuda" or "cpu".

    Returns:
        keep_np: (N,) boolean numpy array indicating which points to keep.
    """
    images_list = list(images.values())
    N = pts_np.shape[0]

    # Move points to device
    pts = torch.from_numpy(pts_np).to(device=device, dtype=torch.float32)  # (N,3)

    # Per-point counters
    total = torch.zeros(N, device=device, dtype=torch.int32)
    outside = torch.zeros(N, device=device, dtype=torch.int32)

    # Move all masks to device once
    mask_cache = {
        img_id: torch.from_numpy(mask_np).to(device=device)
        for img_id, mask_np in mask_cache_np.items()
    }

    print(f"[Torch] Filtering {N} points on device: {device}")
    for img in tqdm(images_list, desc="Images (Torch)"):
        cam = cameras[img.camera_id]
        mask = mask_cache[img.id]  # (H,W) on device

        H, W = mask.shape

        # Build R, t on device
        R_np = colmap.qvec2rotmat(img.qvec)  # (3,3)
        t_np = img.tvec.reshape(3, 1)        # (3,1)

        R = torch.from_numpy(R_np).to(device=device, dtype=torch.float32)  # (3,3)
        t = torch.from_numpy(t_np).to(device=device, dtype=torch.float32)  # (3,1)

        # Camera intrinsics: [fx, fy, cx, cy, ...]
        fx, fy, cx, cy = cam.params[0:4]
        fx = torch.tensor(fx, device=device, dtype=torch.float32)
        fy = torch.tensor(fy, device=device, dtype=torch.float32)
        cx = torch.tensor(cx, device=device, dtype=torch.float32)
        cy = torch.tensor(cy, device=device, dtype=torch.float32)

        # Project all points at once: X_cam = R @ X + t, with X: (3,N)
        X = pts.t()              # (3,N)
        X_cam = R @ X + t        # (3,N)

        Z = X_cam[2]             # (N,)
        valid_z = Z > 1e-6       # points in front of camera

        # Avoid division by zero
        Z_safe = torch.where(valid_z, Z, torch.ones_like(Z))

        u = fx * (X_cam[0] / Z_safe) + cx
        v = fy * (X_cam[1] / Z_safe) + cy

        # Pixel coordinates (integer)
        u_int = u.long()
        v_int = v.long()

        # Valid pixel range
        valid_xy = (
            valid_z &
            (u_int >= 0) & (u_int < W) &
            (v_int >= 0) & (v_int < H)
        )

        if not valid_xy.any():
            continue

        idx_valid = torch.nonzero(valid_xy, as_tuple=False).squeeze(-1)  # (K,)

        v_sel = v_int[idx_valid]
        u_sel = u_int[idx_valid]
        mask_vals = mask[v_sel, u_sel]  # (K,)

        # Update per-point statistics
        total[idx_valid] += 1
        outside[idx_valid] += (mask_vals == 0).to(torch.int32)

    # Compute outside ratio
    total_f = total.to(torch.float32)
    outside_f = outside.to(torch.float32)

    valid_points = total_f > 0
    ratio = torch.zeros_like(total_f)
    ratio[valid_points] = outside_f[valid_points] / total_f[valid_points]

    keep = torch.zeros(N, dtype=torch.bool, device=device)
    keep[valid_points] = ratio[valid_points] < outside_threshold

    # Points that are never projected (total=0) will be keep=False by default.

    keep_np = keep.cpu().numpy()
    return keep_np


def filter_gaussians_3dgs_ply(
    ply_path: str,
    sparse_dir: str,
    image_dir: str,
    output_path: str,
    outside_threshold: float = 0.6,
    input_type: str = "auto",
    device: str = "cuda",
):
    """
    3DGS mode:
      - Input: 3DGS PLY
      - Output: filtered 3DGS PLY (same structure, fewer vertices)
    """

    # Load PLY
    ply = PlyData.read(ply_path)
    vertex = ply["vertex"]
    N = len(vertex)

    # Detect PLY type
    if input_type == "auto":
        detected = detect_ply_type(vertex)
        print(f"[INFO] Detected PLY type: {detected} (auto)")
        ply_type = detected
    else:
        ply_type = input_type
        print(f"[INFO] Using user-specified PLY type: {ply_type}")

    if ply_type != "3dgs":
        print("[WARN] PLY does not look like 3DGS, but proceeding anyway.")

    # Ensure xyz exists
    if not all(name in vertex.data.dtype.names for name in ("x", "y", "z")):
        raise RuntimeError(
            f"PLY does not contain x,y,z properties: {vertex.data.dtype.names}"
        )

    # Extract xyz
    pts = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1)
    print(f"Loaded {N} points from {ply_type} PLY: {ply_path}")

    # Load COLMAP cameras & images
    cameras, images, _ = load_colmap_model(sparse_dir)
    images_list = list(images.values())
    print(f"Loaded {len(images_list)} COLMAP images from: {sparse_dir}")

    # Load alpha masks (on CPU, then moved to GPU in filter_gaussians_torch)
    mask_cache_np = {}
    for img in images_list:
        img_path = os.path.join(image_dir, img.name)
        rgba = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

        if rgba is None or rgba.shape[2] != 4:
            raise RuntimeError(
                f"Cannot read RGBA image (4 channels required): {img_path}"
            )

        alpha = rgba[:, :, 3]
        _, mask = cv2.threshold(alpha, 0, 255, cv2.THRESH_BINARY)
        mask_cache_np[img.id] = mask

    # Decide device
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA not available, falling back to CPU.")
        device = "cpu"

    # Run Torch-based filtering
    keep = filter_gaussians_torch(
        pts_np=pts,
        cameras=cameras,
        images=images,
        mask_cache_np=mask_cache_np,
        outside_threshold=outside_threshold,
        device=device,
    )

    kept_indices = np.where(keep)[0]
    new_vertices = vertex[kept_indices]

    print(f"Remaining: {len(new_vertices)} / {N} points")

    # Re-create PLY with same structure, fewer vertices
    new_ply = PlyData(
        [PlyElement.describe(new_vertices, "vertex")],
        text=ply.text,
    )
    new_ply.write(output_path)
    print("Saved PLY:", output_path)


def filter_colmap_points(
    sparse_dir: str,
    image_dir: str,
    output_sparse_dir: str,
    outside_threshold: float = 0.6,
    device: str = "cuda",
):
    """
    COLMAP mode:
      - Input: COLMAP sparse model (cameras.bin, images.bin, points3D.bin)
      - Output: filtered COLMAP sparse model (same .bin format, fewer points3D)

    This keeps cameras and images unchanged and only filters points3D,
    so you get a 'cleaned' points3D.bin that can be used for 3DGS training.
    """

    # Load COLMAP model
    cameras, images, points3D = load_colmap_model(sparse_dir)
    images_list = list(images.values())
    print(f"[COLMAP] Loaded {len(images_list)} images and {len(points3D)} points from: {sparse_dir}")

    # Build xyz array from points3D
    point_ids = list(points3D.keys())
    pts = np.stack([points3D[pid].xyz for pid in point_ids], axis=0)
    N = pts.shape[0]
    print(f"[COLMAP] Total points3D: {N}")

    # Load alpha masks for all images
    mask_cache_np = {}
    for img in images_list:
        img_path = os.path.join(image_dir, img.name)
        rgba = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

        if rgba is None or rgba.shape[2] != 4:
            raise RuntimeError(
                f"Cannot read RGBA image (4 channels required): {img_path}"
            )

        alpha = rgba[:, :, 3]
        _, mask = cv2.threshold(alpha, 0, 255, cv2.THRESH_BINARY)
        mask_cache_np[img.id] = mask

    # Decide device
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA not available, falling back to CPU.")
        device = "cpu"

    # Run Torch-based filtering on COLMAP points
    keep = filter_gaussians_torch(
        pts_np=pts,
        cameras=cameras,
        images=images,
        mask_cache_np=mask_cache_np,
        outside_threshold=outside_threshold,
        device=device,
    )

    # Subset points3D dict
    kept_ids = [pid for pid, k in zip(point_ids, keep) if k]
    new_points3D = {pid: points3D[pid] for pid in kept_ids}
    print(f"[COLMAP] Remaining points3D: {len(new_points3D)} / {len(points3D)}")

    # Write out new COLMAP model (same cameras & images, filtered points3D)
    os.makedirs(output_sparse_dir, exist_ok=True)
    colmap.write_model(
        cameras=cameras,
        images=images,
        points3D=new_points3D,
        path=output_sparse_dir,
        ext=".bin",
    )
    print(f"[COLMAP] Saved filtered model to: {output_sparse_dir}")
    print("  - cameras.bin")
    print("  - images.bin")
    print("  - points3D.bin")


def main():
    parser = argparse.ArgumentParser(
        description="Filter Gaussian / point cloud using RGBA alpha masks (Torch GPU). "
                    "Supports 3DGS PLY and COLMAP .bin models."
    )

    parser.add_argument(
        "--ply_path",
        type=str,
        # default="/home/yifan/studium/master_thesis/Anomaly_Detection/3dfm4anomaly_detection/scripts/experiment_MAD_Sim_vggt_3dgs_fixCenter_withPoseError_withGhosting/01Gorilla/output/point_cloud/iteration_30000/point_cloud.ply",
        help="Path to input 3DGS .ply file (used only when input_type='3dgs').",
    )

    parser.add_argument(
        "--sparse_dir",
        type=str,
        required=True,
        # default="/home/yifan/studium/master_thesis/Anomaly_Detection/3dfm4anomaly_detection/scripts/experiment_MAD_Sim_vggt_3dgs_fixCenter_withPoseError_withGhosting/01Gorilla/distorted/sparse/0",
        help="COLMAP sparse model directory (cameras.bin, images.bin, points3D.bin).",
    )

    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        # default="/home/yifan/studium/master_thesis/Anomaly_Detection/3dfm4anomaly_detection/scripts/experiment_MAD_Sim_vggt_3dgs_fixCenter_withPoseError_withGhosting/01Gorilla/images",
        help="Directory containing RGBA images (alpha used as mask).",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        # default="/home/yifan/studium/master_thesis/Anomaly_Detection/3dfm4anomaly_detection/scripts/experiment_MAD_Sim_vggt_3dgs_fixCenter_withPoseError_withGhosting/01Gorilla/output/point_cloud/iteration_30000/point_cloud_clean_threshold0_1_gpu.ply",
        help=(
            "If input_type='3dgs': path to save filtered PLY.\n"
            "If input_type='colmap': directory to save filtered COLMAP model "
            "(cameras.bin, images.bin, points3D.bin)."
        ),
    )

    parser.add_argument(
        "--outside_threshold",
        type=float,
        default=0.1,
        help="Max allowed outside-projection ratio per point "
             "(points with ratio >= threshold are removed).",
    )

    parser.add_argument(
        "--input_type",
        type=str,
        default="3dgs",
        choices=["3dgs", "colmap"],
        help=(
            "'3dgs'  = input is 3DGS PLY and output is filtered PLY.\n"
            "'colmap'= input is COLMAP sparse model (.bin) and output is filtered .bin model."
        ),
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run filtering on (default: cuda, falls back to cpu if not available).",
    )

    args = parser.parse_args()

    if args.input_type == "3dgs":
        filter_gaussians_3dgs_ply(
            ply_path=args.ply_path,
            sparse_dir=args.sparse_dir,
            image_dir=args.image_dir,
            output_path=args.output_path,
            outside_threshold=args.outside_threshold,
            input_type="3dgs",
            device=args.device,
        )
    elif args.input_type == "colmap":
        filter_colmap_points(
            sparse_dir=args.sparse_dir,
            image_dir=args.image_dir,
            output_sparse_dir=args.output_path,
            outside_threshold=args.outside_threshold,
            device=args.device,
        )
    else:
        raise ValueError(f"Unknown input_type: {args.input_type}")


if __name__ == "__main__":
    main()

'''
examples:

python remove_pcd_noise_gpu.py \
  --input_type 3dgs \
  --ply_path /path/to/point_cloud.ply \
  --sparse_dir /path/to/distorted/sparse/0 \
  --image_dir /path/to/images \
  --output_path /path/to/point_cloud_clean.ply \
  --outside_threshold 0.1

python remove_pcd_noise_gpu.py \
  --input_type colmap \
  --sparse_dir /path/to/distorted/sparse/0 \
  --image_dir /path/to/images \
  --output_path /path/to/distorted/sparse_clean \
  --outside_threshold 0.1

'''
