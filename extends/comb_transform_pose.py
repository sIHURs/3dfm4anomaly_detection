import json, os, numpy as np
import argparse

# -------- Pose utilities --------
def get_key(frame_path):
    """根据文件名取键，你也可以改成自己的规则。"""
    return os.path.splitext(os.path.basename(frame_path))[0]

def decompose_pose(T, is_c2w=True):
    """
    从 4x4 位姿矩阵取相机中心 C 和 c2w 旋转 R_c2w。
    - c2w:  T = [R | C; 0 0 0 1]
    - w2c:  T = [R | t],  C = -R^T t,  R_c2w = R^T
    """
    R = T[:3,:3]
    t = T[:3,3]
    if is_c2w:
        C = t
        R_c2w = R
    else:
        C = -R.T @ t
        R_c2w = R.T
    return C, R_c2w

def compose_c2w(R_c2w, C):
    T = np.eye(4)
    T[:3,:3] = R_c2w
    T[:3,3]  = C
    return T

def umeyama_align(Y, X):
    """
    行向量版 Umeyama：对齐 Y->X，返回 (s,R,t) 使  X ≈ s * (Y @ R^T) + t
    X,Y: (N,3) 相机中心
    """
    muX, muY = X.mean(0), Y.mean(0)
    X0, Y0 = X - muX, Y - muY
    U, S, Vt = np.linalg.svd((Y0).T @ (X0) / X.shape[0])
    D = np.diag([1,1,np.sign(np.linalg.det(U @ Vt))])
    R = U @ D @ Vt
    varY = (Y0**2).sum() / X.shape[0]
    s = np.trace(np.diag(S) @ D) / varY
    t = muX - s * (Y @ R.T).mean(0) + s * (muY @ R.T)  # 等价写法：muX - s*(R@muY)
    # 为避免混淆，直接返回标准形式
    t = muX - s * (R @ muY)
    return s, R, t

def apply_sim3_to_centers_and_rot(C, R_c2w, s, Rg, tg):
    """
    把相机中心/朝向从子坐标系变换到母坐标系：
    C' = s*(C @ Rg^T) + t
    R' = Rg @ R
    """
    C_new = (s * (C @ Rg.T)) + tg
    R_new = Rg @ R_c2w
    return C_new, R_new

# -------- Main merge --------
def merge_pose_json(json_A, json_B, out_json,
                    is_c2w=True,  # True: transform_matrix 是 c2w；False: 输入是 w2c
                    prefer='A'    # 重复键时保留哪组：'A' or 'B'
                   ):
    A = json.load(open(json_A, 'r'))
    B = json.load(open(json_B, 'r'))

    frames_A = {get_key(f["file_path"]): f for f in A["frames"]}
    frames_B = {get_key(f["file_path"]): f for f in B["frames"]}

    keys_A = set(frames_A.keys())
    keys_B = set(frames_B.keys())
    keys_common = sorted(list(keys_A & keys_B))
    if len(keys_common) < 3:
        raise ValueError(f"公共帧太少（{len(keys_common)}），无法稳定估计 Sim(3)。请确保至少 3 个且分布广。")

    # 提取公共帧的相机中心与旋转（c2w）
    C_A, R_A, C_B, R_B = [], [], [], []
    for k in keys_common:
        Ta = np.array(frames_A[k]["transform_matrix"], dtype=float)
        Tb = np.array(frames_B[k]["transform_matrix"], dtype=float)
        Ca, Ra = decompose_pose(Ta, is_c2w=is_c2w)
        Cb, Rb = decompose_pose(Tb, is_c2w=is_c2w)
        C_A.append(Ca); R_A.append(Ra); C_B.append(Cb); R_B.append(Rb)
    C_A = np.vstack(C_A)  # (N,3)
    C_B = np.vstack(C_B)

    # 估计 B->A 的相似变换
    s, Rg, tg = umeyama_align(C_B, C_A)  # 让 B 对齐到 A
    # 可打印查看
    print(f"[Sim3] scale={s:.6f}\n[Sim3] Rg=\n{Rg}\n[Sim3] t={tg}")

    # 变换 B 的所有帧
    frames_B_aligned = {}
    for k, f in frames_B.items():
        T = np.array(f["transform_matrix"], dtype=float)
        Cb, Rb = decompose_pose(T, is_c2w=is_c2w)
        Cb2, Rb2 = apply_sim3_to_centers_and_rot(Cb, Rb, s, Rg, tg)
        T_new = compose_c2w(Rb2, Cb2)
        f_new = dict(f)
        f_new["transform_matrix"] = T_new.tolist()
        frames_B_aligned[k] = f_new

    # 合并（同键冲突时按 prefer）
    merged = {}
    if prefer.upper() == 'A':
        merged.update(frames_B_aligned)
        merged.update(frames_A)
    else:
        merged.update(frames_A)
        merged.update(frames_B_aligned)

    # 以键排序写回（也可自定义顺序）
    merged_frames = [merged[k] for k in sorted(merged.keys())]

    # 头部元信息：优先沿用 A
    out = dict(A)
    out["frames"] = merged_frames

    with open(out_json, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"✅ merged poses saved to: {out_json}")

# ---------------- Usage example ----------------
# merge_pose_json("part1_transforms.json", "part2_transforms.json", "merged_transforms.json",
#                 is_c2w=True, prefer='A')

def main():
    ap = argparse.ArgumentParser(description="Align and merge two pose JSONs (Sim3 Umeyama).")
    ap.add_argument("--inA", required=True, help="Path to first transforms.json (reference)")
    ap.add_argument("--inB", required=True, help="Path to second transforms.json (to be aligned to A)")
    ap.add_argument("--out", required=True, help="Output merged transforms.json")
    ap.add_argument("--is_c2w", action="store_true", help="Set if inputs are c2w (NeRF/Blender). If omitted, treat as w2c.")
    ap.add_argument("--prefer", default="A", choices=["A","B"], help="Keep A or B on key conflicts (default: A)")
    args = ap.parse_args()

    merge_pose_json(
        json_A=args.inA,
        json_B=args.inB,
        out_json=args.out,
        is_c2w=args.is_c2w,
        prefer=args.prefer
    )

if __name__ == "__main__":
    main()