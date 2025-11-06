import os
import numpy as np
import colmap
out_sparse_dir = "/home/yifan/studium/master_thesis/VGGT-X/vggt_reconstruction/MAD-Sim_full/MAD-Sim_vggt_colmap_for3dgsInput/01Gorilla_shiftedCenter/sparse/0"
in_sparse_dir  = "/home/yifan/studium/master_thesis/VGGT-X/vggt_reconstruction/MAD-Sim_full/MAD-Sim_vggt_colmap_for3dgsInput/01Gorilla/sparse/0"
os.makedirs(out_sparse_dir, exist_ok=True)

cameras, images, points3D = colmap.read_model(in_sparse_dir, ext=".bin")

R_list, t_list, C_list, img_ids = [], [], [], []
for img_id, img in images.items():
    R = colmap.qvec2rotmat(img.qvec)   # world->cam
    t = img.tvec                       # (3,)
    C = -R.T @ t                       # camera center in world
    R_list.append(R)
    t_list.append(t)
    C_list.append(C)
    img_ids.append(img_id)

R_arr = np.stack(R_list, 0)            # (M,3,3)
t_arr = np.stack(t_list, 0)            # (M,3)
C_arr = np.stack(C_list, 0)            # (M,3)

# ====== collect points, calc points center ======
if len(points3D) > 0:
    points_arr = np.array([p.xyz for p in points3D.values()], dtype=np.float64)  # (N,3)
    T = points_arr.mean(axis=0)
    center_source = "points3D centroid"
else:
    points_arr = None
    T = C_arr.mean(axis=0)
    center_source = "camera-centers mean"

print(f"[Centering] Using {center_source}: T = {T}")

if points_arr is not None:
    points_new = points_arr - T

# ====== calc new camera poses ======
# R unchanged；t' = t + R*T
# calc R*T in batch
RT = (R_arr @ T.reshape(3,1)).squeeze(-1)   # (M,3)
t_new = t_arr + RT                          # (M,3)

C_new = -np.transpose(R_arr, (0,2,1)) @ t_new[..., None]  # (M,3,1)
C_new = C_new.squeeze(-1)                                 # (M,3)
assert np.allclose(C_new, C_arr - T, atol=1e-6), "[Check] Camera centers not shifted correctly!"


for i, img_id in enumerate(img_ids):
    images[img_id].tvec = t_new[i].astype(np.float64)

if points_arr is not None:
    for (pid, p), new_xyz in zip(points3D.items(), points_new):
        p.xyz = new_xyz.astype(np.float64)
        points3D[pid] = p

colmap.write_model(cameras, images, points3D, out_sparse_dir, ext=".bin")

print(f"[Done] Written centered model to: {out_sparse_dir}")
