import argparse
import numpy as np
import os
from colmap import qvec2rotmat, rotmat2qvec, read_model, write_model, Point3D
from plyfile import PlyData, PlyElement
from robust_sim3_module import robust_weighted_estimate_sim3_torch


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

    verts["x"], verts["y"], verts["z"] = xyz[:,0], xyz[:,1], xyz[:,2]
    verts["red"], verts["green"], verts["blue"] = rgb[:,0], rgb[:,1], rgb[:,2]

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

    print("DEBUG")
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
            t'_wc = t_wc - R_wc R_align^T t_align
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


def align_and_merge_colmap_models(
    model_A_path,
    model_B_path,
    output_model_path,
    overlap_num=None,
    min_common_for_sim3=3,
):
    """
    使用 overlap 区域的相机中心估计 SE(3): B -> A，并把 B 的点云和相机一起对齐并合并到 A 中。

    参数：
        model_A_path: 第一个 COLMAP 模型目录 (含 cameras.bin/images.bin/points3D.bin)
        model_B_path: 第二个 COLMAP 模型目录
        output_model_path: 输出目录（会创建）
        overlap_num: 重叠帧数量。如果为 None，则使用所有公共图像名。
        min_common_for_sim3: 至少需要多少公共图像才能稳定估计变换
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
    print("Common image names (used for alignment):")
    for n in common_names:
        print("  ", n)

    if len(common_names) < min_common_for_sim3:
        raise ValueError(
            f"公共图像数量太少 ({len(common_names)}), 不能稳定估计变换，"
            f"请检查 overlap_num 或确保两个模型确实有重叠图像。"
        )

    # 4. 构造对应点对 (camera centers)
    src = []  # B 模型的相机中心
    tgt = []  # A 模型的相机中心

    for name in common_names:
        src.append(centers_B[name])
        tgt.append(centers_A[name])

    src = np.stack(src, axis=0).astype(np.float32)  # (N,3)
    tgt = np.stack(tgt, axis=0).astype(np.float32)  # (N,3)
    init_weights = np.ones(src.shape[0], dtype=np.float32)

    print(f"Using {src.shape[0]} camera-center correspondences for SE(3) estimation.")

    # 5. 用 robust_weighted_estimate_sim3_torch，但 align_method='se3'，忽略 scale
    print("Estimating SE(3) with robust_weighted_estimate_sim3_torch (align_method='se3') ...")
    s_dummy, R_align, t_align = robust_weighted_estimate_sim3_torch(
        src=src,
        tgt=tgt,
        init_weights=init_weights,
        delta=0.1,
        max_iters=10000,
        tol=1e-9,
        align_method="se3",   # 关键：只求 R, t
    )
    print("Estimated rotation R_align:\n", R_align)
    print("Estimated translation t_align:", t_align)

    # 6. 对 B 的相机外参做变换，得到 images_B_aligned（在 A 的世界坐标系）
    images_B_aligned = apply_se3_to_cameras(images_B, R_align, t_align)

    # 7. 对 B 的点云做变换，并合并到 A 的 points3D 里
    merged_points3D = {}

    # 7.1 先复制 A 的 points3D，保持原 ID，不动 images_A 里的 tracks
    for pid, pt in points3D_A.items():
        merged_points3D[pid] = pt

    max_id_A = max(merged_points3D.keys()) if merged_points3D else 0
    base_id_offset = max_id_A + 1

    # 7.2 把 B 的点云对齐到 A 的坐标系，重新分配 ID，清空 track
    aligned_points3D_B = apply_se3_to_points3D(points3D_B, R_align, t_align, base_id_offset)
    merged_points3D.update(aligned_points3D_B)

    # ----------- Save PLY files -----------
    ply_dir = os.path.join(output_model_path, "ply")
    os.makedirs(ply_dir, exist_ok=True)

    # A 点云
    xyz_A = np.stack([pt.xyz for pt in points3D_A.values()])
    rgb_A = np.stack([pt.rgb for pt in points3D_A.values()])
    save_points_to_ply(os.path.join(ply_dir, "A_original.ply"), xyz_A, rgb_A)

    # B 对齐前
    xyz_B_raw = np.stack([pt.xyz for pt in points3D_B.values()])
    rgb_B_raw = np.stack([pt.rgb for pt in points3D_B.values()])
    save_points_to_ply(os.path.join(ply_dir, "B_before_align.ply"), xyz_B_raw, rgb_B_raw)

    # B 对齐后
    xyz_B_aligned = np.stack([pt.xyz for pt in aligned_points3D_B.values()])
    rgb_B_aligned = np.stack([pt.rgb for pt in aligned_points3D_B.values()])
    save_points_to_ply(os.path.join(ply_dir, "B_after_align.ply"), xyz_B_aligned, rgb_B_aligned)

    # 合并后的（A + B_aligned）
    xyz_merged = np.concatenate([xyz_A, xyz_B_aligned], axis=0)
    rgb_merged = np.concatenate([rgb_A, rgb_B_aligned], axis=0)
    save_points_to_ply(os.path.join(ply_dir, "merged_AplusB.ply"), xyz_merged, rgb_merged)

    print(f"Merged points3D count: {len(merged_points3D)} (A: {len(points3D_A)}, B aligned: {len(aligned_points3D_B)})")

    # 8. 把 A 的 cameras + images 和 B_aligned 的 images 合并起来
    #   - 简单做法: 只保留 A 的 cameras，假设两个重建使用相同的内参模型；
    #   - images: 为了避免 id 冲突，给 B 的 image id 做一个 offset。
    merged_cameras = cameras_A

    merged_images = {}
    # 8.1 先复制 A 的 images
    for img_id, img in images_A.items():
        merged_images[img_id] = img
    max_img_id_A = max(merged_images.keys()) if merged_images else 0
    img_offset = max_img_id_A

    # 8.2 再把 B_aligned 的 images 加进来，id 重新编号
    for img_id, img in images_B_aligned.items():
        new_id = img_offset + img_id  # 简单偏移
        new_img = type(img)(
            id=new_id,
            qvec=img.qvec,
            tvec=img.tvec,
            camera_id=img.camera_id,  # 这里假设 A/B 使用相同相机模型
            name=img.name,
            xys=img.xys,
            point3D_ids=img.point3D_ids,  # 注意：我们没有让 B 的点云和 2D track 对应起来
        )
        merged_images[new_id] = new_img

    print(f"Merged images count: {len(merged_images)}")

    # 9. 写回一个新的 COLMAP 模型
    write_model(merged_cameras, merged_images, merged_points3D, path=output_model_path, ext=".bin")
    print(f"Aligned + merged COLMAP model (with cameras) has been saved to: {output_model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_A", type=str, default="scripts/test_MAD-Sim_vggt_3dgs_grouped4_align/group1/distorted/sparse/0",
                        help="Path to first COLMAP model (e.g. sparse/0)")
    parser.add_argument("--model_B", type=str, default="scripts/test_MAD-Sim_vggt_3dgs_grouped4_align/group2/distorted/sparse/0",
                        help="Path to second COLMAP model")
    parser.add_argument("--output_model", type=str, default="scripts/test_MAD-Sim_vggt_3dgs_grouped4_align/merged_group12_with_cams",
                        help="Output directory for merged model")
    parser.add_argument("--overlap_num", type=int, default=10,
                        help="Number of overlapping frames (use all common images if not set)")
    args = parser.parse_args()

    align_and_merge_colmap_models(
        model_A_path=args.model_A,
        model_B_path=args.model_B,
        output_model_path=args.output_model,
        overlap_num=args.overlap_num,
    )
