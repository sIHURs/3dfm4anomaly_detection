import os
import json
import math
import argparse
import numpy as np

from colmap import qvec2rotmat, read_model


def make_c2w_from_colmap(R_wc: np.ndarray, t_wc: np.ndarray) -> np.ndarray:
    """Convert COLMAP world->cam extrinsics (R_wc, t_wc) to cam->world 4x4."""
    T = np.eye(4, dtype=float)
    T[:3, :3] = R_wc.T
    T[:3, 3] = (-R_wc.T @ t_wc.reshape(3, 1)).ravel()
    return T

def make_c2w_from_c2w(R_c2w: np.ndarray, t_c2w: np.ndarray) -> np.ndarray:
    """Assemble cam->world 4x4 from (R_c2w, t_c2w)."""
    T = np.eye(4, dtype=float)
    T[:3, :3] = R_c2w
    T[:3, 3] = t_c2w.reshape(3)
    return T

def opencv_to_opengl(T_c2w: np.ndarray) -> np.ndarray:
    """OpenCV camera coords -> common NeRF/Blender OpenGL-like convention."""
    fix = np.diag([1, -1, -1, 1]).astype(float)
    return T_c2w @ fix


def unpack_intrinsics(cam) -> tuple[float, float, float, float]:
    model = str(cam.model)
    if model == "PINHOLE":
        fx, fy, cx, cy = cam.params[:4]
        return float(fx), float(fy), float(cx), float(cy)
    if model == "SIMPLE_PINHOLE":
        f, cx, cy = cam.params[:3]
        fx = fy = float(f)
        return fx, fy, float(cx), float(cy)

    raise RuntimeError(
        f"Only PINHOLE and SIMPLE_PINHOLE are supported, but got camera_id={cam.id} model={cam.model}"
    )


def build_transforms_pinhole(
    cameras,
    images,
    image_base_dir=None,
    sort_by_name=True,
    use_opengl_coords=True,   # 默认 True
):
    # Ensure all cameras are supported
    for cam in cameras.values():
        if str(cam.model) not in ("PINHOLE", "SIMPLE_PINHOLE"):
            raise RuntimeError(
                f"Only PINHOLE and SIMPLE_PINHOLE are supported, but got camera_id={cam.id} model={cam.model}"
            )

    # Use the first camera to compute camera_angle_x
    first_cam = list(cameras.values())[0]
    fx0, fy0, cx0, cy0 = unpack_intrinsics(first_cam)
    angle_x = 2.0 * math.atan(first_cam.width / (2.0 * fx0))
    angle_y = 2.0 * math.atan(first_cam.height / (2.0 * fy0))

    # Sort frames
    img_items = list(images.items())
    img_items = sorted(img_items, key=lambda kv: kv[1].name) if sort_by_name else sorted(img_items, key=lambda kv: kv[0])

    frames = []
    for _, im in img_items:
        R_wc = qvec2rotmat(im.qvec)
        t_wc = im.tvec

        T_c2w = make_c2w_from_colmap(R_wc, t_wc)
        if use_opengl_coords:
            T_c2w = opencv_to_opengl(T_c2w)

        file_path = im.name
        if image_base_dir is not None:
            file_path = os.path.join(image_base_dir, im.name).replace("\\", "/")

        cam = cameras[im.camera_id]
        fx, fy, cx, cy = unpack_intrinsics(cam)

        frames.append(
            {
                "file_path": file_path,
                "transform_matrix": T_c2w.tolist(),
                "fl_x": float(fx),
                "fl_y": float(fy),
                "cx": float(cx),
                "cy": float(cy),
                "w": int(cam.width),
                "h": int(cam.height),
                "camera_model": str(cam.model),
            }
        )

    return {"camera_angle_x": float(angle_x), "camera_angle_y": float(angle_y), "frames": frames}


def write_failed_transforms(out_path: str, reason: str, extra: dict | None = None):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    payload = {
        "reconstruction_failed": True,
        "reason": reason,
        "frames": [],
    }
    if extra:
        payload.update(extra)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"[FAIL] wrote failure transforms.json -> {out_path} | reason: {reason}")


def list_recon_subdirs(sparse_dir: str) -> list[str]:
    """
    Return sorted list of reconstruction subfolder names under sparse_dir
    (only directories, ignoring files).
    """
    if not os.path.isdir(sparse_dir):
        return []
    subs = []
    for name in os.listdir(sparse_dir):
        p = os.path.join(sparse_dir, name)
        if os.path.isdir(p):
            subs.append(name)
    # sort numerically if possible
    def key_fn(x):
        try:
            return (0, int(x))
        except:
            return (1, x)
    return sorted(subs, key=key_fn)


def process_one_scene(scene_dir: str, ext: str, image_base_dir: str | None, sort_by_name: bool, use_opengl_coords: bool):
    """
    scene_dir: e.g. /data/root/01Gorilla
    expects sparse/0 inside scene_dir
    writes transforms.json into scene_dir/transforms.json
    """
    out_path = os.path.join(scene_dir, "transforms_anomaly_free_poses.json")
    sparse_dir = os.path.join(scene_dir, "sparse")
    recons = list_recon_subdirs(sparse_dir)

    if not recons:
        write_failed_transforms(
            out_path,
            reason="missing_sparse_or_no_reconstruction",
            extra={"sparse_dir": "sparse", "found_recon_dirs": recons},
        )
        return

    # 需求：如果 sparse 中有多个重建（不只 0 还有 1/2），视为失败
    if set(recons) != {"0"}:
        write_failed_transforms(
            out_path,
            reason="multiple_or_nonzero_reconstructions",
            extra={"sparse_dir": "sparse", "found_recon_dirs": recons},
        )
        return

    model_dir = os.path.join(sparse_dir, "0")

    try:
        cameras, images, points3D = read_model(model_dir, ext=ext)
        tf = build_transforms_pinhole(
            cameras=cameras,
            images=images,
            image_base_dir=image_base_dir,
            sort_by_name=sort_by_name,
            use_opengl_coords=use_opengl_coords,
        )
    except Exception as e:
        write_failed_transforms(
            out_path,
            reason="exception_during_conversion",
            extra={"error": str(e), "colmap_model": os.path.join("sparse", "0"), "ext": ext},
        )
        return

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(tf, f, indent=2)
    print(f"[OK] {scene_dir} -> {out_path}  (frames={len(tf['frames'])})")


def main():
    ap = argparse.ArgumentParser("Batch COLMAP sparse/0 -> transforms.json (PINHOLE/SIMPLE_PINHOLE)")

    ap.add_argument(
        "--root_dir",
        required=True,
        help="Root folder that contains class subfolders (01Gorilla, 02Unicorn, ...).",
    )

    # 1) 默认 ext 为 .bin
    ap.add_argument("--ext", default=".bin", choices=[".bin", ".txt"], help="COLMAP model file extension (default: .bin)")

    ap.add_argument("--image_base_dir", default=None, help="Prefix for frame file_path, e.g. 'images'")
    ap.add_argument("--sort_by_name", action="store_true", help="Sort frames by filename")

    # 1) 默认开启 use_opengl_coords（用 store_false 让你能关闭）
    ap.add_argument(
        "--no_opengl_coords",
        action="store_true",
        help="Disable OpenCV->OpenGL coordinate fix (default: enabled)",
    )

    ap.add_argument(
        "--only_dirs",
        action="store_true",
        help="Only process direct subdirectories of root_dir (default: process direct subdirs anyway; this flag is kept for clarity).",
    )

    args = ap.parse_args()

    root_dir = args.root_dir
    use_opengl_coords = not args.no_opengl_coords

    if not os.path.isdir(root_dir):
        raise SystemExit(f"root_dir is not a directory: {root_dir}")

    # 遍历 root_dir 的一级子目录（classes）
    subdirs = []
    for name in sorted(os.listdir(root_dir)):
        p = os.path.join(root_dir, name)
        if os.path.isdir(p):
            subdirs.append(p)

    if not subdirs:
        raise SystemExit(f"No subdirectories found under root_dir: {root_dir}")

    print(f"[INFO] root_dir={root_dir}")
    print(f"[INFO] ext={args.ext} | use_opengl_coords={use_opengl_coords} | image_base_dir={args.image_base_dir}")
    print(f"[INFO] found {len(subdirs)} class folders")

    for scene_dir in subdirs:
        process_one_scene(
            scene_dir=scene_dir,
            ext=args.ext,
            image_base_dir=args.image_base_dir,
            sort_by_name=args.sort_by_name,
            use_opengl_coords=use_opengl_coords,
        )


if __name__ == "__main__":
    main()
