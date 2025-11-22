#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''test 1'''
# """
# Align and merge multiple COLMAP models (cameras.bin / images.bin / points3D.bin)
# using ONLY the classes from read_write_model.py (Camera / Image / Point3D).

# Strategy:
# 1) Choose one model as reference (default: model 0).
# 2) For each remaining model:
#    - Estimate Sim(3) from shared image names using camera centers (Umeyama).
#    - Apply Sim(3) to all images (R,t via C) and to all 3D points.
#    - Remap IDs (camera_id, image_id, point3D_id) to avoid collisions.
#    - Merge dictionaries.
# 3) Write merged model to output directory.

# Author: you ;)
# """

# import os
# import sys
# import numpy as np
# from collections import OrderedDict

# # --- 如果 read_write_model.py 不在当前目录，请取消注释并改路径 ---
# # sys.path.append("/path/to/colmap/scripts/python")  # 包含 read_write_model.py 的目录

# import colmap
# # 常见接口：colmap.read_model, colmap.write_model, colmap.Camera, colmap.Image, colmap.Point3D
# # 以及 colmap.qvec2rotmat, colmap.rotmat2qvec


# # ------------------------- 数学与工具函数 -------------------------

# def umeyama(X, Y, with_scale=True):
#     """
#     Estimate Sim(3) (s,R,t) that maps Y -> X:  X ≈ s * R * Y + t
#     X, Y: (N,3)
#     """
#     X = np.asarray(X, dtype=np.float64)
#     Y = np.asarray(Y, dtype=np.float64)
#     assert X.shape == Y.shape and X.ndim == 2 and X.shape[1] == 3
#     n = X.shape[0]
#     mx, my = X.mean(0), Y.mean(0)
#     X0, Y0 = X - mx, Y - my
#     cov = (Y0.T @ X0) / n
#     U, S, Vt = np.linalg.svd(cov)
#     R = U @ Vt
#     if np.linalg.det(R) < 0:
#         U[:, -1] *= -1
#         R = U @ Vt
#         S[-1] *= -1
#     if with_scale:
#         varY = (Y0 ** 2).sum() / n
#         s = S.sum() / varY
#     else:
#         s = 1.0
#     t = mx - s * R @ my
#     return s, R, t


# def image_center_and_RT(image_obj):
#     """
#     Given a read_write_model.Image, return (C, R, t)
#     R: world->cam, t: world->cam, C: camera center in world
#     """
#     R = colmap.qvec2rotmat(image_obj.qvec)
#     t = image_obj.tvec.reshape(3)
#     C = -R.T @ t
#     return C, R, t


# def apply_sim3_to_image(image_obj, s, Rg, tg):
#     """
#     Update image_obj (in place) according to world transform:
#     X' = s * Rg * X + tg

#     To keep projection invariant:
#       C' = s*Rg*C + tg
#       R' = R * Rg^T
#       t' = - R' * C'
#     """
#     C, R, t = image_center_and_RT(image_obj)
#     Cp = s * (Rg @ C) + tg
#     Rp = R @ Rg.T
#     tp = -Rp @ Cp

#     image_obj.qvec = colmap.rotmat2qvec(Rp)
#     image_obj.tvec = tp.astype(np.float64)

#     return image_obj


# def apply_sim3_to_points(points3D_dict, s, Rg, tg):
#     """
#     Transform 3D points in place: X' = s * Rg * X + tg
#     """
#     for pid, P in points3D_dict.items():
#         X = P.xyz
#         Xp = s * (Rg @ X) + tg
#         P.xyz = Xp.astype(np.float64)
#     return points3D_dict


# def transform_model_inplace(cameras, images, points3D, s, Rg, tg):
#     """
#     Apply Sim(3) to whole model (images & points). Cameras unchanged (intrinsics).
#     """
#     # Images
#     for k in images:
#         apply_sim3_to_image(images[k], s, Rg, tg)
#     # Points
#     apply_sim3_to_points(points3D, s, Rg, tg)
#     return cameras, images, points3D


# def read_model_dir(model_dir):
#     """
#     Return (cameras, images, points3D) as OrderedDicts using read_write_model.py.
#     """
#     cams, imgs, pts = colmap.read_model(model_dir, ext=".bin")
#     # 确保为 OrderedDict（read_model 通常已是）
#     return OrderedDict(cams), OrderedDict(imgs), OrderedDict(pts)


# def write_model_dir(model_tuple, out_dir):
#     os.makedirs(out_dir, exist_ok=True)
#     colmap.write_model(*model_tuple, out_dir, ext=".bin")


# def common_name_pairs(imgs_ref, imgs_mov):
#     """
#     Build pairs (ref_img_id, mov_img_id) by matching Image.name.
#     """
#     name_to_id_ref = {}
#     for iid, I in imgs_ref.items():
#         name_to_id_ref[I.name] = iid

#     pairs = []
#     for iid2, J in imgs_mov.items():
#         nm = J.name
#         if nm in name_to_id_ref:
#             pairs.append((name_to_id_ref[nm], iid2))
#     return pairs


# def sim3_from_shared_cameras(imgs_ref, imgs_mov):
#     """
#     Estimate Sim(3) that maps mov -> ref using shared image names and camera centers.
#     Return (s, Rg, tg) for X_ref ≈ s*Rg*X_mov + tg
#     """
#     pairs = common_name_pairs(imgs_ref, imgs_mov)
#     if len(pairs) < 3:
#         raise RuntimeError(f"Shared images too few: {len(pairs)}. Need >= 3 for robust Sim(3).")
#     C_ref, C_mov = [], []
#     for k_ref, k_mov in pairs:
#         C1, _, _ = image_center_and_RT(imgs_ref[k_ref])
#         C2, _, _ = image_center_and_RT(imgs_mov[k_mov])
#         C_ref.append(C1); C_mov.append(C2)
#     C_ref = np.stack(C_ref, 0)
#     C_mov = np.stack(C_mov, 0)
#     s, Rg, tg = umeyama(C_ref, C_mov, with_scale=True)
#     return s, Rg, tg


# # ------------------------- 合并与重编号 -------------------------

# def clone_camera_with_new_id(C, new_id):
#     """
#     Create a NEW Camera instance with given id.
#     read_write_model.Camera fields: id, model, width, height, params
#     """
#     return colmap.Camera(
#         id=new_id,
#         model=C.model,
#         width=C.width,
#         height=C.height,
#         params=C.params.copy()
#     )


# def clone_image_with_new_ids(I, new_image_id, new_camera_id, point3D_id_offset=0):
#     """
#     Create a NEW Image with remapped ids.
#     Image fields: id, qvec, tvec, camera_id, name, xys, point3D_ids
#     """
#     # remap point3D_ids (-1 stays -1)
#     if I.point3D_ids is not None:
#         new_pids = np.array([(pid + point3D_id_offset) if pid != -1 else -1
#                              for pid in I.point3D_ids], dtype=np.int64)
#     else:
#         new_pids = I.point3D_ids

#     return colmap.Image(
#         id=new_image_id,
#         qvec=I.qvec.copy(),
#         tvec=I.tvec.copy(),
#         camera_id=new_camera_id,
#         name=I.name,
#         xys=I.xys.copy() if I.xys is not None else None,
#         point3D_ids=new_pids
#     )


# def clone_point3D_with_new_id(P, new_point_id, image_id_offset=0):
#     """
#     Create a NEW Point3D with remapped id and (optionally) image_ids offset.
#     Point3D fields: id, xyz, rgb, error, image_ids, point2D_idxs
#     """
#     img_ids = P.image_ids
#     if img_ids is not None and len(img_ids) > 0:
#         new_img_ids = np.array([iid + image_id_offset for iid in img_ids], dtype=np.int64)
#     else:
#         new_img_ids = img_ids

#     return colmap.Point3D(
#         id=new_point_id,
#         xyz=P.xyz.copy(),
#         rgb=P.rgb.copy(),
#         error=float(P.error),
#         image_ids=new_img_ids,
#         point2D_idxs=P.point2D_idxs.copy() if P.point2D_idxs is not None else None
#     )


# def next_offsets_from_model(model):
#     cams, imgs, pts = model
#     # max existing ids; if empty, start from 1
#     next_img = (max(imgs.keys()) + 1) if len(imgs) > 0 else 1
#     next_pt  = (max(pts.keys()) + 1) if len(pts)  > 0 else 1
#     next_cam = (max(cams.keys()) + 1) if len(cams) > 0 else 1
#     return next_cam, next_img, next_pt


# def remap_and_merge_models(ref_model, mov_model):
#     """
#     Merge mov_model into ref_model with NEW ids (no collision).
#     Return merged_model.
#     """
#     cams_ref, imgs_ref, pts_ref = ref_model
#     cams_mov, imgs_mov, pts_mov = mov_model

#     # Output dicts (copies)
#     cams_out = OrderedDict(cams_ref)
#     imgs_out = OrderedDict(imgs_ref)
#     pts_out  = OrderedDict(pts_ref)

#     # Decide offsets
#     next_cam_id, next_img_id, next_pt_id = next_offsets_from_model(ref_model)

#     # ---- Cameras: clone with new IDs and build old->new map ----
#     cam_id_map = {}
#     for old_cid, C in cams_mov.items():
#         new_cid = next_cam_id
#         cams_out[new_cid] = clone_camera_with_new_id(C, new_cid)
#         cam_id_map[old_cid] = new_cid
#         next_cam_id += 1

#     # ---- Points: clone with new IDs & offset their track image_ids later after images are merged ----
#     #   We need image_id_offset to fix tracks; but here we're not using a constant offset, we give new image ids sequentially.
#     #   So we'll temporarily store points and fix image_ids after we know image_id mapping.
#     tmp_points_new = {}
#     for old_pid, P in pts_mov.items():
#         # Create placeholder with SAME image_ids for now; we'll remap later using image_id_map
#         tmp_points_new[old_pid] = colmap.Point3D(
#             id=-1,  # placeholder
#             xyz=P.xyz.copy(),
#             rgb=P.rgb.copy(),
#             error=float(P.error),
#             image_ids=P.image_ids.copy() if P.image_ids is not None else None,
#             point2D_idxs=P.point2D_idxs.copy() if P.point2D_idxs is not None else None
#         )

#     # ---- Images: clone with new image_id & remap camera_id; also shift point3D_ids by a CONSTANT pid offset ----
#     #    We'll set a constant PID offset to avoid O(N) map for pids, then later also rebuild points to match that offset.
#     pid_offset = next_pt_id
#     image_id_map = {}
#     for old_iid, I in imgs_mov.items():
#         new_iid = next_img_id
#         image_id_map[old_iid] = new_iid
#         next_img_id += 1

#         new_cam_id = cam_id_map[I.camera_id]
#         # For point3D_ids inside Image, we add CONSTANT offset: new_pid = old_pid + pid_offset (keep -1)
#         new_I = clone_image_with_new_ids(I, new_image_id=new_iid, new_camera_id=new_cam_id,
#                                          point3D_id_offset=pid_offset)
#         imgs_out[new_iid] = new_I

#     # ---- Now finalize Points3D: assign new ids = old_pid + pid_offset, and remap their track image_ids via image_id_map ----
#     for old_pid, Ptemp in tmp_points_new.items():
#         new_pid = old_pid + pid_offset

#         if Ptemp.image_ids is not None and len(Ptemp.image_ids) > 0:
#             new_img_ids = np.array([image_id_map[iid] for iid in Ptemp.image_ids], dtype=np.int64)
#         else:
#             new_img_ids = Ptemp.image_ids

#         new_P = colmap.Point3D(
#             id=new_pid,
#             xyz=Ptemp.xyz,
#             rgb=Ptemp.rgb,
#             error=Ptemp.error,
#             image_ids=new_img_ids,
#             point2D_idxs=Ptemp.point2D_idxs
#         )
#         pts_out[new_pid] = new_P

#     return cams_out, imgs_out, pts_out


# # ------------------------- 主流程：对齐 + 合并 -------------------------

# def align_and_merge(model_dirs, out_dir, ref_idx=0):
#     """
#     model_dirs: list of directories each containing cameras.bin/images.bin/points3D.bin
#     out_dir: output sparse directory (e.g., ".../merged/sparse/0")
#     ref_idx: which model is the reference frame (0-based)
#     """
#     assert len(model_dirs) >= 2, "Need at least 2 models."
#     models = [read_model_dir(d) for d in model_dirs]

#     # Set reference model
#     ref_model = models[ref_idx]
#     merged = ref_model  # will accumulate

#     # Merge others one by one
#     for i, mov_model_orig in enumerate(models):
#         if i == ref_idx:
#             continue

#         # --- copy to avoid touching original dicts ---
#         cams_mov = OrderedDict((cid, colmap.Camera(id=C.id, model=C.model, width=C.width,
#                                                    height=C.height, params=C.params.copy()))
#                                for cid, C in mov_model_orig[0].items())

#         imgs_mov = OrderedDict((iid, colmap.Image(
#                                     id=I.id, qvec=I.qvec.copy(), tvec=I.tvec.copy(),
#                                     camera_id=I.camera_id, name=I.name,
#                                     xys=I.xys.copy() if I.xys is not None else None,
#                                     point3D_ids=I.point3D_ids.copy() if I.point3D_ids is not None else None))
#                                for iid, I in mov_model_orig[1].items())

#         pts_mov = OrderedDict((pid, colmap.Point3D(
#                                     id=P.id, xyz=P.xyz.copy(), rgb=P.rgb.copy(),
#                                     error=float(P.error),
#                                     image_ids=P.image_ids.copy() if P.image_ids is not None else None,
#                                     point2D_idxs=P.point2D_idxs.copy() if P.point2D_idxs is not None else None))
#                               for pid, P in mov_model_orig[2].items())

#         mov_model = (cams_mov, imgs_mov, pts_mov)

#         # 1) Estimate Sim(3): mov -> merged (use shared image names)
#         s, Rg, tg = sim3_from_shared_cameras(merged[1], mov_model[1])

#         # 2) Transform mov model into merged coordinate (in place)
#         transform_model_inplace(mov_model[0], mov_model[1], mov_model[2], s, Rg, tg)

#         # 3) Remap IDs & merge
#         merged = remap_and_merge_models(merged, mov_model)

#     # Write output
#     write_model_dir(merged, out_dir)
#     print(f"✅ Merged {len(model_dirs)} models. Output -> {out_dir}")
#     return merged


# # ------------------------- CLI -------------------------

# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser(description="Align and merge multiple COLMAP models using read_write_model.py classes.")
#     parser.add_argument("--models", nargs="+", required=True,
#                         help="List of model directories (each contains cameras.bin/images.bin/points3D.bin).")
#     parser.add_argument("--out", required=True, help="Output directory for merged model (e.g., .../merged/sparse/0).")
#     parser.add_argument("--ref_idx", type=int, default=0, help="Index of reference model in --models (default 0).")
#     args = parser.parse_args()

#     # Example:
#     # python merge_colmap_readwrite.py \
#     #   --models /path/g1/sparse/0 /path/g2/sparse/0 /path/g3/sparse/0 /path/g4/sparse/0 \
#     #   --out /path/merged/sparse/0 --ref_idx 0

#     align_and_merge(args.models, args.out, ref_idx=args.ref_idx)
'''test 2, translation'''
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Merge two COLMAP models by TRANSLATION ONLY, based on shared image names:
- Compute average camera center over shared images in each model.
- delta = mean(C1_shared) - mean(C2_shared)
- Translate ALL images & points of model2 by delta.
- Merge: drop the duplicated shared images from model2.
- Remap IDs to avoid collisions.
- Save as COLMAP (read_write_model.py).
"""

import os
import sys
import numpy as np
from collections import OrderedDict

# 如果 read_write_model.py 不在同目录，请自行添加路径
# sys.path.append("/path/to/colmap/scripts/python")
import colmap

# ---------- 基础工具 ----------

def read_model_dir(model_dir):
    cams, imgs, pts = colmap.read_model(model_dir, ext=".bin")
    return OrderedDict(cams), OrderedDict(imgs), OrderedDict(pts)

def write_model_dir(model, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    colmap.write_model(*model, out_dir, ext=".bin")

def image_center_and_RT(I):
    R = colmap.qvec2rotmat(I.qvec)   # world->cam
    t = I.tvec.reshape(3)            # world->cam
    C = -R.T @ t                     # camera center in world
    return C, R, t

def apply_translation_to_image(I, delta):
    """
    世界坐标做平移: X' = X + delta
    ==> 相机中心 C' = C + delta, 旋转不变, t' = -R * C'
    """
    C, R, t = image_center_and_RT(I)
    Cp = C + delta
    tp = -R @ Cp
    I.tvec = tp.astype(np.float64)
    # 旋转不变，qvec不改
    return I

def apply_translation_to_points(points3D, delta):
    for pid, P in points3D.items():
        P.xyz = (P.xyz + delta).astype(np.float64)
    return points3D

def common_name_pairs(imgs1, imgs2):
    name2id_1 = {}
    for iid, I in imgs1.items():
        name2id_1[I.name] = iid
    pairs = []
    for iid2, J in imgs2.items():
        if J.name in name2id_1:
            pairs.append((name2id_1[J.name], iid2))
    return pairs

def mean_center_over_pairs(imgs, idxs):
    Cs = []
    for iid in idxs:
        C, _, _ = image_center_and_RT(imgs[iid])
        Cs.append(C)
    if len(Cs) == 0:
        raise RuntimeError("No indices to average.")
    return np.mean(np.stack(Cs, 0), axis=0)


# ---------- 克隆与重映射（使用 read_write_model.py 的类） ----------

def clone_camera_with_new_id(C, new_id):
    return colmap.Camera(
        id=new_id,
        model=C.model,
        width=C.width,
        height=C.height,
        params=C.params.copy()
    )

def clone_image_with_new_ids(I, new_image_id, new_camera_id, point3D_id_offset=0):
    if I.point3D_ids is not None:
        new_pids = np.array([(pid + point3D_id_offset) if pid != -1 else -1
                             for pid in I.point3D_ids], dtype=np.int64)
    else:
        new_pids = None
    return colmap.Image(
        id=new_image_id,
        qvec=I.qvec.copy(),
        tvec=I.tvec.copy(),
        camera_id=new_camera_id,
        name=I.name,
        xys=I.xys.copy() if I.xys is not None else None,
        point3D_ids=new_pids
    )

def clone_point3D_with_new_id(P, new_point_id, image_id_map=None):
    if P.image_ids is not None and len(P.image_ids) > 0:
        if image_id_map is None:
            new_img_ids = P.image_ids.copy()
        else:
            # 仅保留映射成功（被保留的图像）的轨迹
            kept = []
            for iid in P.image_ids:
                if iid in image_id_map:
                    kept.append(image_id_map[iid])
            new_img_ids = np.array(kept, dtype=np.int64) if len(kept) > 0 else np.zeros((0,), dtype=np.int64)
    else:
        new_img_ids = P.image_ids
    return colmap.Point3D(
        id=new_point_id,
        xyz=P.xyz.copy(),
        rgb=P.rgb.copy(),
        error=float(P.error),
        image_ids=new_img_ids,
        point2D_idxs=P.point2D_idxs.copy() if P.point2D_idxs is not None else None
    )

def next_offsets_from_model(model):
    cams, imgs, pts = model
    next_cam = (max(cams.keys()) + 1) if len(cams) > 0 else 1
    next_img = (max(imgs.keys()) + 1) if len(imgs) > 0 else 1
    next_pt  = (max(pts.keys())  + 1) if len(pts)  > 0 else 1
    return next_cam, next_img, next_pt


# ---------- 主逻辑：基于共享图像的平移对齐 + 合并（去重共享图像） ----------

def translate_and_merge(model1_dir, model2_dir, out_dir, verbose=True):
    cams1, imgs1, pts1 = read_model_dir(model1_dir)
    cams2, imgs2, pts2 = read_model_dir(model2_dir)

    # 1) 找共享图像
    pairs = common_name_pairs(imgs1, imgs2)
    if len(pairs) == 0:
        raise RuntimeError("两个模型没有共享的 view（按 Image.name 匹配）。无法按该思路平移。")

    shared_iids_1 = [i1 for i1, _ in pairs]
    shared_iids_2 = [i2 for _, i2 in pairs]

    # 2) 计算两侧共享相机中心的平均值
    meanC1 = mean_center_over_pairs(imgs1, shared_iids_1)
    meanC2 = mean_center_over_pairs(imgs2, shared_iids_2)

    # 3) 平移量：delta = meanC1 - meanC2
    delta = meanC1 - meanC2
    if verbose:
        print(f"[Info] #shared views = {len(pairs)}")
        print(f"[Info] meanC1 = {meanC1}, meanC2 = {meanC2}, delta = {delta}")

    # 4) 将 model2 的所有相机与点云平移
    #    注意：旋转不变，t = -R*C'（由 apply_translation_to_image 完成）
    for iid in imgs2:
        apply_translation_to_image(imgs2[iid], delta)
    apply_translation_to_points(pts2, delta)

    # 5) 合并：删除 model2 里的共享图像，仅保留非共享的
    shared_names = set([imgs2[i2].name for _, i2 in pairs])
    keep_iids2 = [iid for iid, I in imgs2.items() if I.name not in shared_names]

    # 6) 复制 model1 作为起点
    cams_out = OrderedDict(cams1)
    imgs_out = OrderedDict(imgs1)
    pts_out  = OrderedDict(pts1)

    # 7) 计算各类 ID 的起始偏移
    next_cam_id, next_img_id, next_pt_id = next_offsets_from_model((cams_out, imgs_out, pts_out))

    # 8) 为 model2 的相机分配新 ID，建映射
    cam_id_map = {}
    for old_cid, C in cams2.items():
        new_cid = next_cam_id
        cams_out[new_cid] = clone_camera_with_new_id(C, new_cid)
        cam_id_map[old_cid] = new_cid
        next_cam_id += 1

    # 9) 先为 model2（保留的图像）计算 image_id 映射
    image_id_map = {}
    for old_iid in keep_iids2:
        new_iid = next_img_id
        image_id_map[old_iid] = new_iid
        next_img_id += 1

    # 10) points3D 的新 ID 采用常量偏移：new_pid = old_pid + pid_offset
    pid_offset = next_pt_id

    # 11) 写入保留的 images（来自 model2，非共享）
    for old_iid in keep_iids2:
        I = imgs2[old_iid]
        new_iid = image_id_map[old_iid]
        new_cam_id = cam_id_map[I.camera_id]
        new_I = clone_image_with_new_ids(I, new_image_id=new_iid,
                                         new_camera_id=new_cam_id,
                                         point3D_id_offset=pid_offset)
        imgs_out[new_iid] = new_I

    # 12) 写入 model2 的 points3D（全部），并将其 track 的 image_ids 映射到新的 image_id；
    #     若某点只被共享图像观测，而这些图像被丢弃，则该点会变成“无轨迹点”，依然可以保留其坐标与颜色。
    for old_pid, P in pts2.items():
        new_pid = old_pid + pid_offset
        new_P = clone_point3D_with_new_id(P, new_point_id=new_pid, image_id_map=image_id_map)
        pts_out[new_pid] = new_P

    # 13) 写合并结果
    write_model_dir((cams_out, imgs_out, pts_out), out_dir)
    print(f"✅ Done. Merged model saved to: {out_dir}")
    print(f"   Cameras: {len(cams_out)}, Images: {len(imgs_out)}, Points3D: {len(pts_out)}")


# ---------- CLI ----------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Translate-align (by shared views' mean center) and merge two COLMAP models.")
    parser.add_argument("--model1", required=True, help="Path to model1 directory (contains cameras.bin/images.bin/points3D.bin)")
    parser.add_argument("--model2", required=True, help="Path to model2 directory (contains cameras.bin/images.bin/points3D.bin)")
    parser.add_argument("--out", required=True, help="Output directory for merged model (e.g., .../merged/sparse/0)")
    args = parser.parse_args()

    translate_and_merge(args.model1, args.model2, args.out, verbose=True)
