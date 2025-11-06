import os, json
import numpy as np
import colmap as rw  # 来自 COLMAP scripts/python
from pathlib import Path
from analyis_transforms_pose import umeyama_align

# ---------- 工具函数 ----------
def umeyama_align(Y, X):
    """
    求 Y->X 的相似变换：X ≈ s * Rg @ Y + tg
    Y, X: (N,3)
    返回: s (float), Rg (3,3), tg (3,)
    """
    muX, muY = X.mean(0), Y.mean(0)
    X0, Y0 = X - muX, Y - muY
    U, S, Vt = np.linalg.svd(Y0.T @ X0 / Y.shape[0])
    R = U @ np.diag([1,1,np.sign(np.linalg.det(U @ Vt))]) @ Vt
    varY = (Y0**2).sum() / Y.shape[0]
    s = np.trace(np.diag(S) @ np.diag([1,1,np.sign(np.linalg.det(U @ Vt))])) / varY
    t = muX - s * (R @ muY)
    return float(s), R, t

def c2w_to_colmap_qt(T_c2w):
    """将 c2w（4x4）转为 COLMAP 的 world->cam: (qvec, tvec)"""
    Rcw = T_c2w[:3,:3]
    tcw = T_c2w[:3, 3]
    Rwc = Rcw.T
    twc = - Rwc @ tcw
    qvec = rw.rotmat2qvec(Rwc)
    return qvec, twc

def image_center_from_qt(qvec, tvec):
    """由 COLMAP 的 world->cam (R,t) 计算相机中心 C（world系）"""
    R = rw.qvec2rotmat(qvec)
    t = tvec
    C = - R.T @ t
    return C

def apply_sim3_to_point(X, s, Rg, tg):
    return s * (Rg @ X) + tg

def apply_sim3_to_image(qvec_c2w, tvec_c2w, s, Rg, tg):
    # """对 COLMAP 外参施加世界系的 Sim(3)"""
    # Rcw = rw.qvec2rotmat(qvec_c2w)        # c2w 旋转
    # tcw = np.asarray(tvec_c2w, dtype=float)

    # Rcw_p = Rg @ Rcw                       # 旋转左乘 Rg
    # tcw_p = s * (Rg @ tcw) + tg            # 平移缩放+旋转+平移

    # qvec_p = rw.rotmat2qvec(Rcw_p)
    

    R  = rw.qvec2rotmat(qvec_c2w)
    t = tvec_c2w

    R, t = make_c2w_from_colmap(R, t)

    t = (s * (t @ Rg.T)) + tg
    # R = np.einsum('ij,njk->nik', Rg, R)
    R = R @ Rg.T

    qvec_p = rw.rotmat2qvec(R)
    tcw_p = t

    return qvec_p, tcw_p

def match_by_suffix(name, keys):
    """用文件名后缀匹配 transforms.json 的 file_path（容错，有时含相对路径）"""
    for k in keys:
        if k.endswith(name):
            return k
    return None

def make_c2w_from_colmap(R_wc, t_wc):
    """world->cam 外参 (R_wc,t_wc) 转 cam->world 4x4"""
    T = np.eye(4, dtype=float)
    T[:3, :3] = R_wc.T
    T[:3, 3]  = (-R_wc.T @ t_wc.reshape(3, 1)).ravel()
    return T[:3, :3], T[:3, 3]

# ---------- 主函数：把 transforms.json 的对齐写回 COLMAP 并同步点云 ----------
def export_aligned_colmap(
    in_sparse_dir, out_sparse_dir,
    T_map=None,           # dict: file_path -> 4x4 c2w (numpy array)
    sim3=None             # (s, Rg, tg) 若你已知 Umeyama 结果，可直接传入
):
    """
    - 若提供 sim3：用它统一变换 images.bin + points3D.bin
    - 否则：用 T_map 与原 images.bin 的相机中心估 sim3，再统一变换
    """
    os.makedirs(out_sparse_dir, exist_ok=True)
    cameras, images, points3D = rw.read_model(in_sparse_dir, ext=".bin")

    # 1) 若无 sim3，则通过 T_map 和原 images 估计
    if sim3 is None:
        assert T_map is not None and len(T_map) > 2, "需要 T_map（且数量>2）来估计 Sim(3)"
        # 收集对应的 C_pred(来自 T_map) 与 C_orig(来自 images.bin)
        C_pred, C_orig = [], []
        for img in images.values():
            key = match_by_suffix(img.name, T_map.keys())
            if key is None: 
                continue
            # 由 T_map 的 c2w 得到中心
            C_aligned = T_map[key][:3, 3]
            # 原模型的中心
            C0 = image_center_from_qt(img.qvec, img.tvec)
            C_pred.append(C_aligned)
            C_orig.append(C0)
        C_pred = np.asarray(C_pred)
        C_orig = np.asarray(C_orig)
        assert len(C_pred) >= 3, "用于估算 Sim(3) 的相机数量不足（<3）"
        s, Rg, tg = umeyama_align(C_pred, C_orig)  # 让对齐后的C_pred 去贴原坐标的 C_orig？还是反之
        # 注意：我们要把“对齐相机 & 点云”写成 **aligned 的新坐标系**。
        # 如果 T_map 是“预测->GT”的结果，你更可能想让**原模型**变到 **T_map 对齐后的坐标**。
        # 这时应反过来估计：   sim3 = align(C_orig -> C_pred)
        # 即：
        s, Rg, tg = umeyama_align(C_orig, C_pred)
    else:

        s, Rg, tg = sim3

    # 2) 统一变换 points3D
    for pid, P in points3D.items():
        P.xyz = apply_sim3_to_point(P.xyz, s, Rg, tg)
        points3D[pid] = P

    # 3) 统一变换 images（外参）
    for iid, I in images.items():
        I.qvec, I.tvec = apply_sim3_to_image(I.qvec, I.tvec, s, Rg, tg)
        images[iid] = I

    # 4) cameras.bin 一般不动（内参与世界尺度无关）
    rw.write_model(cameras, images, points3D, out_sparse_dir, ext=".bin")
    print(f"[OK] Wrote aligned COLMAP model to: {out_sparse_dir}")
    print(f"     (s={s:.6f}, Rg shape={Rg.shape}, tg={tg})")

def load_sim3_from_json(json_path):
    with open(json_path, "r") as f:
        d = json.load(f)

    s  = d.get("scale", d.get("s"))
    Rg = d.get("rotation", d.get("R", d.get("Rg")))
    tg = d.get("translation", d.get("t", d.get("tg")))

    if s is None or Rg is None or tg is None:
        raise ValueError(f"Invalid SIM(3) json: {json_path}. Required keys: scale/rotation/translation")

    s  = float(s)
    Rg = np.asarray(Rg, dtype=float)
    tg = np.asarray(tg, dtype=float).reshape(3)

    # 基本校验
    if Rg.shape != (3,3):
        raise ValueError(f"rotation must be 3x3, got {Rg.shape}")
    if not np.isfinite([s, *Rg.flatten(), *tg]).all():
        raise ValueError("SIM(3) contains non-finite values")
    return s, Rg, tg


# ---------- 使用示例 ----------
if __name__ == "__main__":

    in_sparse = "/home/yifan/studium/master_thesis/VGGT-X/vggt_reconstruction/MAD-Sim_full/MAD-Sim_vggt_colmap_for3dgsInput/01Gorilla/sparse/0"
    out_sparse = "/home/yifan/studium/master_thesis/VGGT-X/vggt_reconstruction/MAD-Sim_full/MAD-Sim_vggt_colmap_for3dgsInput/01Gorilla_align"

    # 方案A：没有现成的 (s,Rg,tg)，用 T_map 与原 images.bin 的中心估 sim3，然后统一写回 .bin

    sim3 = load_sim3_from_json("/home/yifan/studium/master_thesis/VGGT-X/vggt_reconstruction/MAD-Sim_full/MAD-Sim_vggt_colmap_for3dgsInput/01Gorilla/umeyama_align.json")
    export_aligned_colmap(in_sparse, out_sparse, T_map=None, sim3=sim3)

    # 方案B：如果你手里已有 umeyama 的 (s,Rg,tg)，直接：
    # s, Rg, tg = 1.2345, np.eye(3), np.zeros(3)
    # export_aligned_colmap(in_sparse, out_sparse, sim3=(s,Rg,tg))
