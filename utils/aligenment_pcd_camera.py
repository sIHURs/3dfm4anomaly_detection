#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import os
from colmap import qvec2rotmat, rotmat2qvec, read_model, write_model, Point3D
from plyfile import PlyData, PlyElement
from robust_sim3_module import robust_weighted_estimate_sim3_torch


# =========================
# Utilities
# =========================

def save_points_to_ply(path, xyz, rgb=None):
    """
    Save a point cloud to a colored PLY.

    Args:
        xyz: (N, 3) float
        rgb: (N, 3) uint8 or None (if None, default white)
    """
    xyz = np.asarray(xyz, dtype=np.float32)

    if rgb is None:
        rgb = (np.ones_like(xyz) * 255).astype(np.uint8)
    else:
        rgb = np.asarray(rgb, dtype=np.uint8)

    assert xyz.shape[0] == rgb.shape[0]
    N = xyz.shape[0]

    verts = np.empty(
        N,
        dtype=[
            ("x", "f4"), ("y", "f4"), ("z", "f4"),
            ("red", "u1"), ("green", "u1"), ("blue", "u1"),
        ],
    )
    verts["x"], verts["y"], verts["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    verts["red"], verts["green"], verts["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]

    el = PlyElement.describe(verts, "vertex")
    PlyData([el]).write(path)
    print(f"[PLY SAVED] {path} ({N} points)")


def get_camera_centers_from_images(images):
    """
    Extract camera centers from COLMAP 'images' dict:
        {image_name: C_world (3,)}

    COLMAP convention:
        X_cam = R * X_world + t
        camera center C = -R^T * t
    """
    centers = {}
    for _, img in images.items():
        R_wc = qvec2rotmat(img.qvec)          # world -> camera
        t_wc = img.tvec.reshape(3, 1)         # (3,1)
        C = -R_wc.T @ t_wc                    # (3,1)
        centers[img.name] = C.ravel()         # (3,)
    return centers


def apply_se3_to_xyz(xyz, R_align, t_align):
    """
    Apply SE(3) transform to 3D points.

    Args:
        xyz: (N,3)
        R_align: (3,3)  world_B -> world_A
        t_align: (3,)
    Returns:
        (N,3) transformed points in world_A
    """
    xyz = np.asarray(xyz, dtype=np.float32)
    R_align = np.asarray(R_align, dtype=np.float32)
    t_align = np.asarray(t_align, dtype=np.float32).reshape(1, 3)
    return (xyz @ R_align.T) + t_align


def apply_se3_to_points3D(points3D_B, R_align, t_align, base_id_offset):
    """
    Transform B model points3D with SE(3) and re-assign point IDs to avoid conflicts.
    Tracks (image_ids / point2D_idxs) are cleared to avoid dangling references.
    """
    new_points3D = {}
    cur_id = base_id_offset

    for _, pt in points3D_B.items():
        xyz_new = apply_se3_to_xyz(pt.xyz[None, :], R_align, t_align)[0]

        # Clear tracks to avoid references to non-existing image IDs after merging
        image_ids = np.zeros((0,), dtype=np.int32)
        point2D_idxs = np.zeros((0,), dtype=np.int32)

        new_points3D[cur_id] = Point3D(
            id=cur_id,
            xyz=xyz_new,
            rgb=pt.rgb,
            error=pt.error,
            image_ids=image_ids,
            point2D_idxs=point2D_idxs,
        )
        cur_id += 1

    return new_points3D


def apply_se3_to_cameras(images_B, R_align, t_align):
    """
    Align B camera poses from world_B to world_A.

    Given:
        X_A = R_align X_B + t_align   (world_B -> world_A)
        Original extrinsics: X_cam = R_wc X_B + t_wc

    Derivation:
        X_B = R_align^T (X_A - t_align)
        X_cam = R_wc R_align^T X_A + (t_wc - R_wc R_align^T t_align)

    So:
        R'_wc = R_wc R_align^T
        t'_wc = t_wc - R'_wc t_align
    """
    new_images_B = {}
    R_align = np.asarray(R_align, dtype=np.float32)
    t_align = np.asarray(t_align, dtype=np.float32).reshape(3,)

    for img_id, img in images_B.items():
        R_wc = qvec2rotmat(img.qvec)
        t_wc = img.tvec.reshape(3,)

        R_wc_new = R_wc @ R_align.T
        t_wc_new = t_wc - R_wc_new @ t_align

        qvec_new = rotmat2qvec(R_wc_new)

        new_img = type(img)(
            id=img.id,
            qvec=qvec_new,
            tvec=t_wc_new,
            camera_id=img.camera_id,
            name=img.name,
            xys=img.xys,
            point3D_ids=img.point3D_ids,
        )
        new_images_B[img_id] = new_img

    return new_images_B


# =========================
# Main: SE(3) from overlap cameras + merge cameras, but:
#   1) keep only ONE copy of overlap cameras (by image name)
#   2) keep points only from specified group (A or B)
# =========================

def align_and_merge_colmap_models(
    model_A_path,
    model_B_path,
    output_model_path,
    overlap_num=None,
    min_common_for_sim3=3,
    keep_points="A",  # "A" or "B"
):
    """
    Estimate SE(3) from overlapping camera centers (B -> A), align B cameras, and merge:
      - Keep only one copy of overlapping cameras (drop B images whose names overlap A).
      - Keep points only from the specified group ('A' or 'B').

    Args:
        model_A_path: COLMAP model dir for group A (contains cameras.bin/images.bin/points3D.bin)
        model_B_path: COLMAP model dir for group B
        output_model_path: output directory
        overlap_num: overlap frame count. If None, use all common image names.
        min_common_for_sim3: minimum overlap images required
        keep_points: "A" to keep A points only, "B" to keep (aligned) B points only
    """
    os.makedirs(output_model_path, exist_ok=True)

    cameras_A, images_A, points3D_A = read_model(model_A_path, ext=".bin")
    cameras_B, images_B, points3D_B = read_model(model_B_path, ext=".bin")

    print(f"[Model A] images: {len(images_A)}, points3D: {len(points3D_A)}")
    print(f"[Model B] images: {len(images_B)}, points3D: {len(points3D_B)}")

    centers_A = get_camera_centers_from_images(images_A)
    centers_B = get_camera_centers_from_images(images_B)

    names_A_sorted = sorted(centers_A.keys())
    names_B_sorted = sorted(centers_B.keys())

    if overlap_num is None:
        common_names = sorted(set(names_A_sorted) & set(names_B_sorted))
    else:
        overlap_A = set(names_A_sorted[-overlap_num:])
        overlap_B = set(names_B_sorted[:overlap_num])
        common_names = sorted(overlap_A & overlap_B)

    print(f"Number of candidate common images for alignment: {len(common_names)}")
    print("Common image names (used for alignment):")
    for n in common_names:
        print("  ", n)

    if len(common_names) < min_common_for_sim3:
        raise ValueError(
            f"Too few common images ({len(common_names)}) to estimate a stable transform. "
            f"Check overlap_num or ensure the models share overlapping frames."
        )

    src = np.stack([centers_B[n] for n in common_names], axis=0).astype(np.float32)
    tgt = np.stack([centers_A[n] for n in common_names], axis=0).astype(np.float32)
    init_weights = np.ones(src.shape[0], dtype=np.float32)

    print(f"Using {src.shape[0]} camera-center correspondences for SE(3) estimation.")
    print("Estimating SE(3) with robust_weighted_estimate_sim3_torch (align_method='se3') ...")
    _, R_align, t_align = robust_weighted_estimate_sim3_torch(
        src=src,
        tgt=tgt,
        init_weights=init_weights,
        delta=0.1,
        max_iters=2000,
        tol=1e-9,
        align_method="se3",
    )
    print("Estimated rotation R_align:\n", R_align)
    print("Estimated translation t_align:", t_align)

    # Align B cameras into A world
    images_B_aligned = apply_se3_to_cameras(images_B, R_align, t_align)

    # =========================
    # Points: keep only one group's point cloud
    # =========================
    keep_points = keep_points.upper()
    if keep_points not in ("A", "B"):
        raise ValueError("--keep_points must be 'A' or 'B'")

    merged_points3D = {}
    ply_dir = os.path.join(output_model_path, "ply")
    os.makedirs(ply_dir, exist_ok=True)

    xyz_A = np.stack([pt.xyz for pt in points3D_A.values()]) if len(points3D_A) > 0 else np.zeros((0, 3), np.float32)
    rgb_A = np.stack([pt.rgb for pt in points3D_A.values()]) if len(points3D_A) > 0 else np.zeros((0, 3), np.uint8)

    xyz_B_raw = np.stack([pt.xyz for pt in points3D_B.values()]) if len(points3D_B) > 0 else np.zeros((0, 3), np.float32)
    rgb_B_raw = np.stack([pt.rgb for pt in points3D_B.values()]) if len(points3D_B) > 0 else np.zeros((0, 3), np.uint8)

    # Save diagnostics
    if len(xyz_A) > 0:
        save_points_to_ply(os.path.join(ply_dir, "A_original.ply"), xyz_A, rgb_A)
    if len(xyz_B_raw) > 0:
        save_points_to_ply(os.path.join(ply_dir, "B_before_align.ply"), xyz_B_raw, rgb_B_raw)

    if keep_points == "A":
        # Keep A points only
        for pid, pt in points3D_A.items():
            merged_points3D[pid] = pt
        print("[MERGE] Keeping points3D from A only. Dropping B points3D.")
        if len(xyz_A) > 0:
            save_points_to_ply(os.path.join(ply_dir, "merged_points_A_only.ply"), xyz_A, rgb_A)

    else:
        # Keep (aligned) B points only, in A coordinate system
        # Reassign IDs starting from 1 (or keep original if you prefer)
        base_id_offset = 1
        aligned_points3D_B = apply_se3_to_points3D(points3D_B, R_align, t_align, base_id_offset)
        merged_points3D.update(aligned_points3D_B)
        print("[MERGE] Keeping points3D from B only (aligned to A). Dropping A points3D.")

        if len(aligned_points3D_B) > 0:
            xyz_B_aligned = np.stack([pt.xyz for pt in aligned_points3D_B.values()])
            rgb_B_aligned = np.stack([pt.rgb for pt in aligned_points3D_B.values()])
            save_points_to_ply(os.path.join(ply_dir, "B_after_align.ply"), xyz_B_aligned, rgb_B_aligned)
            save_points_to_ply(os.path.join(ply_dir, "merged_points_B_only_aligned.ply"), xyz_B_aligned, rgb_B_aligned)

    print(f"Merged points3D count: {len(merged_points3D)}")

    # =========================
    # Cameras / images: merge, but keep only ONE copy of overlap images
    # =========================
    merged_cameras = cameras_A  # assume same intrinsics model; adjust if needed

    merged_images = {}

    # Add all A images first
    for img_id, img in images_A.items():
        merged_images[img_id] = img
    max_img_id_A = max(merged_images.keys()) if merged_images else 0

    names_in_A = set(centers_A.keys())  # image names in A
    overlap_set = set(common_names)     # names used for alignment (overlap)

    # Add B images, but:
    #   - drop overlapping names (keep A's copy)
    #   - also drop any name already in A (stronger rule, safe)
    #   - re-id to avoid collisions
    next_id = max_img_id_A + 1
    dropped = 0
    kept = 0

    for _, img in images_B_aligned.items():
        if (img.name in overlap_set) or (img.name in names_in_A):
            dropped += 1
            continue

        new_img = type(img)(
            id=next_id,
            qvec=img.qvec,
            tvec=img.tvec,
            camera_id=img.camera_id,
            name=img.name,
            xys=img.xys,
            point3D_ids=img.point3D_ids,  # not maintained; OK if you only need cameras for 3DGS
        )
        merged_images[next_id] = new_img
        next_id += 1
        kept += 1

    print(f"[MERGE] Images kept from B (non-overlap): {kept}, dropped (overlap/duplicate): {dropped}")
    print(f"Merged images count: {len(merged_images)}")

    write_model(merged_cameras, merged_images, merged_points3D, path=output_model_path, ext=".bin")
    print(f"Aligned + merged COLMAP model saved to: {output_model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_A",
        type=str,
        default="scripts/test_MAD-Sim_vggt_3dgs_grouped4_align/group1/distorted/sparse/0",
        help="Path to first COLMAP model (e.g., sparse/0) [Group A]",
    )
    parser.add_argument(
        "--model_B",
        type=str,
        default="scripts/test_MAD-Sim_vggt_3dgs_grouped4_align/group2/distorted/sparse/0",
        help="Path to second COLMAP model (e.g., sparse/0) [Group B]",
    )
    parser.add_argument(
        "--output_model",
        type=str,
        default="scripts/test_MAD-Sim_vggt_3dgs_grouped4_align/merged_group12_with_cams",
        help="Output directory for merged model",
    )
    parser.add_argument(
        "--overlap_num",
        type=int,
        default=10,
        help="Number of overlapping frames for alignment (A tail vs B head).",
    )
    parser.add_argument(
        "--keep_points",
        type=str,
        default="A",
        choices=["A", "B", "a", "b"],
        help="Keep points3D only from the specified group: 'A' (default) or 'B' (aligned to A).",
    )
    args = parser.parse_args()

    align_and_merge_colmap_models(
        model_A_path=args.model_A,
        model_B_path=args.model_B,
        output_model_path=args.output_model,
        overlap_num=args.overlap_num,
        keep_points=args.keep_points,
    )
