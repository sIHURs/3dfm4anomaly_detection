import os
import json
import math
import argparse
import numpy as np

from colmap import qvec2rotmat, read_model

def make_c2w_from_colmap(R_wc, t_wc):
    """world->cam 外参 (R_wc,t_wc) 转 cam->world 4x4"""
    T = np.eye(4, dtype=float)
    T[:3, :3] = R_wc.T
    T[:3, 3]  = (-R_wc.T @ t_wc.reshape(3, 1)).ravel()
    return T

def make_w2c_from_colmap(R_wc, t_wc):
    """world->cam 外参 (R_wc,t_wc) 转 world->cam 4x4"""
    T = np.eye(4, dtype=float)
    T[:3, :3] = R_wc
    T[:3, 3]  = t_wc.reshape(3, 1).ravel()
    return T

def opencv_to_opengl(T_c2w):
    """右乘固定矩阵，得到 y 上、z 朝后（常见于 NeRF/Blender）的坐标系"""
    fix = np.diag([1, -1, -1, 1]).astype(float)
    return T_c2w @ fix

def build_transforms_pinhole(cameras, images, image_base_dir=None,
                             sort_by_name=True, use_opengl_coords=False):
    # 只允许 PINHOLE
    for cam in cameras.values():
        if str(cam.model) != "PINHOLE":
            raise RuntimeError(
                f"Only PINHOLE is supported, but got camera_id={cam.id} model={cam.model}"
            )

    # 取第一台相机用于 camera_angle_x（NeRF 传统字段）
    first_cam = list(cameras.values())[0]
    fx0, fy0, cx0, cy0 = first_cam.params[:4]
    angle_x = 2.0 * math.atan(first_cam.width / (2.0 * fx0))

    # 帧排序
    img_items = list(images.items())
    img_items = sorted(img_items, key=lambda kv: kv[1].name) if sort_by_name else sorted(img_items, key=lambda kv: kv[0])

    frames = []
    for _, im in img_items:
        # R_wc, t_wc（world->cam）
        R_wc = qvec2rotmat(im.qvec)
        t_wc = im.tvec
        # cam->world
        T_c2w = make_c2w_from_colmap(R_wc, t_wc)
        if use_opengl_coords:
            T_c2w = opencv_to_opengl(T_c2w)

        # file_path
        file_path = im.name
        if image_base_dir is not None:
            file_path = os.path.join(image_base_dir, im.name).replace("\\", "/")

        cam = cameras[im.camera_id]
        fx, fy, cx, cy = cam.params[:4]  # PINHOLE: [fx, fy, cx, cy]

        frames.append({
            "file_path": file_path,
            "transform_matrix": T_c2w.tolist(),
            # 为稳妥，仍把每帧内参写进去（很多管线会忽略，但保留更安全）
            "fl_x": float(fx), "fl_y": float(fy),
            "cx": float(cx),   "cy": float(cy),
            "w": int(cam.width), "h": int(cam.height),
            "camera_model": "PINHOLE"
        })

    transforms = {
        "camera_angle_x": float(angle_x),
        "frames": frames
    }
    return transforms

def main():
    ap = argparse.ArgumentParser("COLMAP (.bin/.txt) -> transforms.json (PINHOLE only)")
    ap.add_argument("--colmap_model", required=True, help="Dir containing cameras.bin, images.bin, (points3D.bin)")
    ap.add_argument("--ext", default=".bin", choices=[".bin", ".txt"], help="Model file extension")
    ap.add_argument("--out", default="transforms.json", help="Output JSON path")
    ap.add_argument("--image_base_dir", default=None, help="Prefix for frame file_path, e.g. 'images'")
    ap.add_argument("--sort_by_name", action="store_true", help="Sort frames by filename")
    ap.add_argument("--use_opengl_coords", action="store_true", help="Apply OpenCV->OpenGL coord fix")
    args = ap.parse_args()

    cameras, images, points3D = read_model(args.colmap_model, ext=args.ext)
    tf = build_transforms_pinhole(
        cameras=cameras,
        images=images,
        image_base_dir=args.image_base_dir,
        sort_by_name=args.sort_by_name,
        use_opengl_coords=args.use_opengl_coords
    )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(tf, f, indent=2)
    print(f"[OK] wrote {args.out} with {len(tf['frames'])} frames (PINHOLE only).")

if __name__ == "__main__":
    main()