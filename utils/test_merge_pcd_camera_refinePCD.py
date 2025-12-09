import argparse
import numpy as np
import os
from colmap import qvec2rotmat, rotmat2qvec, read_model, write_model, Point3D
from plyfile import PlyData, PlyElement
from robust_sim3_module import robust_weighted_estimate_sim3_torch


# =========================
# 工具函数
# =========================

def save_points_to_ply(path, xyz, rgb=None):
    """
    xyz: (N,3)
    rgb: (N,3) or None (if None, default white)
    """
    xyz = np.asarray(xyz, dtype=np.float32)

    if rgb is None:
        rgb = np.ones_like(xyz) * 255
    else:
        rgb = np.asarray(rgb, dtype=np.uint8)

    assert xyz.shape[0] == rgb.shape[0]

    N = xyz.shape[0]

    verts = np.empty(N, dtype=[
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1")
    ])

    verts["x"], verts["y"], verts["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    verts["red"], verts["green"], verts["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]

    el = PlyElement.describe(verts, "vertex")
    PlyData([el]).write(path)

    print(f"[PLY SAVED] {path} ({N} points)")


def get_camera_centers_from_images(images):
    """
    从 COLMAP 的 images 字典中提取相机中心:
        {image_name: C_world (3,)}
    COLMAP 约定:
        X_cam = R * X_world + t
        camera center C = -R^T * t
    """
    centers = {}
    for img_id, img in images.items():
        R_wc = qvec2rotmat(img.qvec)         # world -> camera
        t_wc = img.tvec.reshape(3, 1)        # (3,1)
        C = -R_wc.T @ t_wc                   # (3,1)
        centers[img.name] = C.ravel()        # 存成 (3,)
    return centers


def apply_se3_to_xyz(xyz, R_align, t_align):
    """
    xyz: (N,3)
    R_align: (3,3)  world_B -> world_A
    t_align: (3,)
    返回: (N,3) 对齐后的点云
    """
    xyz = np.asarray(xyz, dtype=np.float32)
    R_align = np.asarray(R_align, dtype=np.float32)
    t_align = np.asarray(t_align, dtype=np.float32).reshape(1, 3)
    return (xyz @ R_align.T) + t_align  # (N,3)


def apply_se3_to_points3D(points3D_B, R_align, t_align, base_id_offset):
    """
    对 B 模型的 points3D 做 SE(3) 变换，并且重新分配 ID（避免和 A 冲突）。
    为了避免 image_ids/point2D_idxs 指向不存在的 images，这里简单把 track 清空。
    """
    new_points3D = {}
    cur_id = base_id_offset

    for old_id, pt in points3D_B.items():
        xyz_new = apply_se3_to_xyz(pt.xyz[None, :], R_align, t_align)[0]  # (1,3) -> (3,)

        # 不保留 B 的 track，避免引用不存在的 image_id
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
    将 B 模型的 camera pose 也从 world_B 对齐到 world_A.
    已知:
        X_A = R_align X_B + t_align   (world_B -> world_A)
        原外参: X_cam = R_wc X_B + t_wc
    推导得到:
        新外参: X_cam = R'_wc X_A + t'_wc
        其中:
            R'_wc = R_wc R_align^T
            t'_wc = t_wc - R'_wc t_align
    """
    new_images_B = {}

    R_align = np.asarray(R_align, dtype=np.float32)
    t_align = np.asarray(t_align, dtype=np.float32).reshape(3,)

    for img_id, img in images_B.items():
        R_wc = qvec2rotmat(img.qvec)                 # (3,3)
        t_wc = img.tvec.reshape(3,)                  # (3,)

        R_wc_new = R_wc @ R_align.T                  # (3,3)
        t_wc_new = t_wc - R_wc_new @ t_align         # (3,)

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


def build_point_correspondences(xyz_A, xyz_B_aligned,
                                num_samples=5000,
                                max_dist_ratio=0.05):
    """
    在“已经用相机中心粗对齐后的” B 点云 (xyz_B_aligned) 和 A 点云之间建立 3D–3D 最近邻对应。
    - xyz_A: (NA,3)
    - xyz_B_aligned: (NB,3)   已经 roughly 对齐到 A 的坐标系

    返回:
      src_pts: (M,3) 来自 B_aligned 的点
      tgt_pts: (M,3) 来自 A 的点
    """
    xyz_A = np.asarray(xyz_A, dtype=np.float32)
    xyz_B_aligned = np.asarray(xyz_B_aligned, dtype=np.float32)

    NA = xyz_A.shape[0]
    NB = xyz_B_aligned.shape[0]
    print(f"[CORR] points_A: {NA}, points_B_aligned: {NB}")

    # 用 A 的 bbox 对角线估一个尺度来做距离阈值
    bbox_min = xyz_A.min(axis=0)
    bbox_max = xyz_A.max(axis=0)
    bbox_diag = np.linalg.norm(bbox_max - bbox_min)
    dist_thresh = bbox_diag * max_dist_ratio
    print(f"[CORR] bbox_diag={bbox_diag:.4f}, max_dist_thresh={dist_thresh:.4f}")

    # 从 B_aligned 中采样
    if NB <= num_samples:
        sample_idx_B = np.arange(NB)
    else:
        sample_idx_B = np.random.choice(NB, size=num_samples, replace=False)

    sampled_B = xyz_B_aligned[sample_idx_B]  # (K,3)

    batch_size = 1024
    src_list = []
    tgt_list = []

    for start in range(0, sampled_B.shape[0], batch_size):
        end = min(start + batch_size, sampled_B.shape[0])
        B_batch = sampled_B[start:end]  # (B,3)

        # 暴力最近邻：dist^2 = |B|^2 + |A|^2 - 2 B·A^T
        B2 = np.sum(B_batch**2, axis=1, keepdims=True)       # (B,1)
        A2 = np.sum(xyz_A**2, axis=1, keepdims=True).T       # (1,NA)
        d2 = B2 + A2 - 2.0 * B_batch @ xyz_A.T               # (B,NA)
        nn_idx = np.argmin(d2, axis=1)                       # (B,)
        nn_d = np.sqrt(np.min(d2, axis=1))                   # (B,)

        mask = nn_d < dist_thresh
        if np.any(mask):
            src_list.append(B_batch[mask])
            tgt_list.append(xyz_A[nn_idx[mask]])

    if len(src_list) == 0:
        raise ValueError("[CORR] No close 3D–3D correspondences found, check overlap or thresholds.")

    src_pts = np.concatenate(src_list, axis=0)
    tgt_pts = np.concatenate(tgt_list, axis=0)
    print(f"[CORR] built {src_pts.shape[0]} correspondences after filtering")

    return src_pts, tgt_pts


# =========================
# 主流程：camera center 粗对齐 + point cloud refine + 合并
# =========================

def align_and_merge_colmap_models(
    model_A_path,
    model_B_path,
    output_model_path,
    overlap_num=None,
    min_common_for_sim3=3,
    num_corr_samples=5000,
    max_corr_dist_ratio=0.05,
):
    """
    1. 使用 overlap 区域的相机中心估计 SE(3): B -> A（粗对齐）；
    2. 用粗对齐后的 B 点云与 A 点云做最近邻对应，再用点云 refine 一次 SE(3)；
    3. 使用最终 SE(3) 对齐 B 的相机位姿和点云，并与 A 合并。
    """

    os.makedirs(output_model_path, exist_ok=True)

    # 1. 读入两个模型
    cameras_A, images_A, points3D_A = read_model(model_A_path, ext=".bin")
    cameras_B, images_B, points3D_B = read_model(model_B_path, ext=".bin")

    print(f"[Model A] images: {len(images_A)}, points3D: {len(points3D_A)}")
    print(f"[Model B] images: {len(images_B)}, points3D: {len(points3D_B)}")

    # 2. 计算相机中心
    centers_A = get_camera_centers_from_images(images_A)
    centers_B = get_camera_centers_from_images(images_B)

    # 3. 根据 image.name 匹配公共图像（不依赖 image_id）
    names_A_sorted = sorted(centers_A.keys())
    names_B_sorted = sorted(centers_B.keys())

    if overlap_num is None:
        # 使用所有公共图像
        common_names = sorted(set(names_A_sorted) & set(names_B_sorted))
    else:
        # 使用 A 末尾 overlap_num 张，和 B 开头 overlap_num 张的交集
        overlap_A = set(names_A_sorted[-overlap_num:])
        overlap_B = set(names_B_sorted[:overlap_num])
        common_names = sorted(overlap_A & overlap_B)

    print(f"Number of candidate common images for alignment: {len(common_names)}")
    print("Common image names (used for camera-center alignment):")
    for n in common_names:
        print("  ", n)

    if len(common_names) < min_common_for_sim3:
        raise ValueError(
            f"公共图像数量太少 ({len(common_names)}), 不能稳定估计变换，"
            f"请检查 overlap_num 或确保两个模型确实有重叠图像。"
        )

    # 4. 构造 camera center 对应 (B -> A)
    src_centers = []  # B 模型的相机中心
    tgt_centers = []  # A 模型的相机中心

    for name in common_names:
        src_centers.append(centers_B[name])
        tgt_centers.append(centers_A[name])

    src_centers = np.stack(src_centers, axis=0).astype(np.float32)  # (N,3)
    tgt_centers = np.stack(tgt_centers, axis=0).astype(np.float32)  # (N,3)
    init_weights = np.ones(src_centers.shape[0], dtype=np.float32)

    print(f"Using {src_centers.shape[0]} camera-center correspondences for SE(3) estimation.")

    # 5. 用 robust_weighted_estimate_sim3_torch 粗估 SE(3): B -> A
    print("Estimating initial SE(3) with robust_weighted_estimate_sim3_torch (align_method='se3') ...")
    s_dummy, R_align, t_align = robust_weighted_estimate_sim3_torch(
        src=src_centers,
        tgt=tgt_centers,
        init_weights=init_weights,
        delta=0.1,
        max_iters=10000,   # 这里不用 10000，避免太慢；camera center 通常收敛很快
        tol=1e-9,
        align_method="se3",
    )
    print("Initial rotation R_align:\n", R_align)
    print("Initial translation t_align:", t_align)
    print(f"s_dummy (should be ~1 in se3 mode): {s_dummy}")

    # 6. 点云 refine：先用 R_align, t_align 把 B 点云粗对齐，然后用最近邻做二次 SE(3) 拟合
    xyz_A = np.stack([pt.xyz for pt in points3D_A.values()])
    xyz_B_raw = np.stack([pt.xyz for pt in points3D_B.values()])

    # B 粗对齐到 A
    xyz_B_initial = apply_se3_to_xyz(xyz_B_raw, R_align, t_align)

    print("\n[Refine] Building point correspondences for point-cloud-based refinement ...")
    src_pts, tgt_pts = build_point_correspondences(
        xyz_A=xyz_A,
        xyz_B_aligned=xyz_B_initial,
        num_samples=num_corr_samples,
        max_dist_ratio=max_corr_dist_ratio,
    )

    refine_weights = np.ones(src_pts.shape[0], dtype=np.float32)
    print(f"[Refine] Using {src_pts.shape[0]} 3D–3D correspondences for refinement.")

    print("[Refine] Estimating refinement SE(3) on top of initial alignment ...")
    s_dummy2, R_refine, t_refine = robust_weighted_estimate_sim3_torch(
        src=src_pts.astype(np.float32),
        tgt=tgt_pts.astype(np.float32),
        init_weights=refine_weights,
        delta=0.1,
        max_iters=2000,
        tol=1e-9,
        align_method="se3",
    )
    print("Refined rotation R_refine:\n", R_refine)
    print("Refined translation t_refine:", t_refine)
    print(f"s_dummy2 (should be ~1 in se3 mode): {s_dummy2}")

    # 组合变换：原始 B 坐标 -> 粗对齐后 -> refine 后
    # X_A_final = R_refine * (R_align X_B + t_align) + t_refine
    R_final = R_refine @ R_align
    t_final = R_refine @ t_align + t_refine

    print("\n[Final] Combined rotation R_final:\n", R_final)
    print("[Final] Combined translation t_final:", t_final)

    # 7. 基于最终 R_final, t_final 对 B 的相机和点云做对齐
    images_B_aligned = apply_se3_to_cameras(images_B, R_final, t_final)

    merged_points3D = {}
    # 7.1 复制 A 的 points3D
    for pid, pt in points3D_A.items():
        merged_points3D[pid] = pt

    max_id_A = max(merged_points3D.keys()) if merged_points3D else 0
    base_id_offset = max_id_A + 1

    # 7.2 B 的点云用最终变换对齐
    aligned_points3D_B = apply_se3_to_points3D(points3D_B, R_final, t_final, base_id_offset)
    merged_points3D.update(aligned_points3D_B)

    # ----------- Save PLY files -----------
    ply_dir = os.path.join(output_model_path, "ply")
    os.makedirs(ply_dir, exist_ok=True)

    # A 点云
    rgb_A = np.stack([pt.rgb for pt in points3D_A.values()])
    save_points_to_ply(os.path.join(ply_dir, "A_original.ply"), xyz_A, rgb_A)

    # B 对齐前
    rgb_B_raw = np.stack([pt.rgb for pt in points3D_B.values()])
    save_points_to_ply(os.path.join(ply_dir, "B_before_align.ply"), xyz_B_raw, rgb_B_raw)

    # B camera-center 初始对齐后的点云
    save_points_to_ply(os.path.join(ply_dir, "B_after_initial_align.ply"),
                       xyz_B_initial, rgb_B_raw)

    # B refine + 最终对齐后的点云（通过 aligned_points3D_B）
    xyz_B_final = np.stack([pt.xyz for pt in aligned_points3D_B.values()])
    rgb_B_final = np.stack([pt.rgb for pt in aligned_points3D_B.values()])
    save_points_to_ply(os.path.join(ply_dir, "B_after_refine_align.ply"),
                       xyz_B_final, rgb_B_final)

    # 合并后的（A + B_final）
    xyz_merged = np.concatenate([xyz_A, xyz_B_final], axis=0)
    rgb_merged = np.concatenate([rgb_A, rgb_B_final], axis=0)
    save_points_to_ply(os.path.join(ply_dir, "merged_AplusB_final.ply"),
                       xyz_merged, rgb_merged)

    print(f"Merged points3D count: {len(merged_points3D)} "
          f"(A: {len(points3D_A)}, B aligned: {len(aligned_points3D_B)})")

    # 8. 合并 cameras + images
    merged_cameras = cameras_A

    merged_images = {}
    # 8.1 复制 A 的 images
    for img_id, img in images_A.items():
        merged_images[img_id] = img
    max_img_id_A = max(merged_images.keys()) if merged_images else 0
    img_offset = max_img_id_A

    # 8.2 加入 B_aligned 的 images（id 做一个 offset）
    for img_id, img in images_B_aligned.items():
        new_id = img_offset + img_id  # 简单偏移
        new_img = type(img)(
            id=new_id,
            qvec=img.qvec,
            tvec=img.tvec,
            camera_id=img.camera_id,  # 假设 A/B 使用相同的 camera model & 内参
            name=img.name,
            xys=img.xys,
            point3D_ids=img.point3D_ids,  # 我们没有维护 B 点云的 track 关系
        )
        merged_images[new_id] = new_img

    print(f"Merged images count: {len(merged_images)}")

    # 9. 写回一个新的 COLMAP 模型
    write_model(merged_cameras, merged_images, merged_points3D, path=output_model_path, ext=".bin")
    print(f"Aligned + merged COLMAP model (with cameras, refined by points) has been saved to: {output_model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_A",
        type=str,
        default="scripts/test_MAD-Sim_vggt_3dgs_grouped4_align/group1/distorted/sparse/0",
        help="Path to first COLMAP model (e.g. sparse/0)",
    )
    parser.add_argument(
        "--model_B",
        type=str,
        default="scripts/test_MAD-Sim_vggt_3dgs_grouped4_align/group2/distorted/sparse/0",
        help="Path to second COLMAP model",
    )
    parser.add_argument(
        "--output_model",
        type=str,
        default="scripts/test_MAD-Sim_vggt_3dgs_grouped4_align/merged_group12_with_cams_refine",
        help="Output directory for merged model",
    )
    parser.add_argument(
        "--overlap_num",
        type=int,
        default=10,
        help="Number of overlapping frames (use all common images if not set)",
    )
    parser.add_argument(
        "--num_corr_samples",
        type=int,
        default=5000,
        help="Number of points sampled from B (after initial alignment) for point-cloud refinement",
    )
    parser.add_argument(
        "--max_corr_dist_ratio",
        type=float,
        default=0.1,
        help="Max NN distance for point correspondences as fraction of A bbox diagonal",
    )
    args = parser.parse_args()

    align_and_merge_colmap_models(
        model_A_path=args.model_A,
        model_B_path=args.model_B,
        output_model_path=args.output_model,
        overlap_num=args.overlap_num,
        num_corr_samples=args.num_corr_samples,
        max_corr_dist_ratio=args.max_corr_dist_ratio,
    )
