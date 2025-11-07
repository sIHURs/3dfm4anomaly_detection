import os
from typing import Literal, Optional, Dict, Any
import argparse
import numpy as np
import colmap


def recenter_colmap_model(
    in_sparse_dir: str,
    out_sparse_dir: str,
    center_by: Literal["auto", "points", "cameras"] = "auto",
    overwrite: bool = True,
) -> Dict[str, Any]:
    """
    Recenter a COLMAP sparse model by shifting either:
      - the 3D points centroid to the origin, or
      - the mean of camera centers to the origin.

    The transform applied is: x' = x - T  (world coords)
    For camera extrinsics (world->cam): R' = R (unchanged), t' = t + R*T

    Args:
        in_sparse_dir:  Path to input sparse/0 directory with cameras.bin, images.bin, points3D.bin (points may be absent).
        out_sparse_dir: Path to output sparse/0 directory to write the recentered model.
        center_by:      "auto" (prefer points if available), or force "points" / "cameras".
        overwrite:      If True, create/overwrite the output directory.

    Returns:
        A dict with useful info:
            {
              "T": np.ndarray shape (3,), the shift applied,
              "used": "points3D centroid" or "camera-centers mean",
              "num_images": int,
              "num_points": int
            }
    """

    in_sparse_dir = os.path.join(in_sparse_dir, "sparse", "0")
    out_sparse_dir = os.path.join(out_sparse_dir, "sparse", "0")
    
    # --- I/O prep ---
    if not os.path.isdir(in_sparse_dir):
        raise FileNotFoundError(f"Input sparse dir not found: {in_sparse_dir}")

    if overwrite:
        os.makedirs(out_sparse_dir, exist_ok=True)
    else:
        if os.path.exists(out_sparse_dir) and os.listdir(out_sparse_dir):
            raise FileExistsError(f"Output dir exists and is not empty: {out_sparse_dir}")


    # --- Load model ---
    cameras, images, points3D = colmap.read_model(in_sparse_dir, ext=".bin")

    # --- Collect camera extrinsics and centers ---
    R_list, t_list, C_list, img_ids = [], [], [], []
    for img_id, img in images.items():
        R = colmap.qvec2rotmat(img.qvec)     # world->cam
        t = img.tvec                         # (3,)
        C = -R.T @ t                         # camera center in world
        R_list.append(R)
        t_list.append(t)
        C_list.append(C)
        img_ids.append(img_id)

    if len(img_ids) == 0:
        raise RuntimeError("No images found in the COLMAP model.")

    R_arr = np.stack(R_list, 0)              # (M,3,3)
    t_arr = np.stack(t_list, 0)              # (M,3)
    C_arr = np.stack(C_list, 0)              # (M,3)

    # --- Determine center T ---
    use_points = False
    points_arr: Optional[np.ndarray] = None

    if center_by in ("auto", "points") and len(points3D) > 0:
        points_arr = np.array([p.xyz for p in points3D.values()], dtype=np.float64)  # (N,3)
        T = points_arr.mean(axis=0)
        used = "points3D centroid"
        use_points = True
    elif center_by == "points" and len(points3D) == 0:
        raise RuntimeError("center_by='points' requested but points3D.bin is missing or empty.")
    else:
        T = C_arr.mean(axis=0)
        used = "camera-centers mean"

    # --- Shift points (if present) ---
    if use_points and points_arr is not None:
        points_new = points_arr - T
        # write back
        for (pid, p), new_xyz in zip(points3D.items(), points_new):
            p.xyz = new_xyz.astype(np.float64)
            points3D[pid] = p

    # --- Update camera extrinsics: R unchanged; t' = t + R*T ---
    RT = (R_arr @ T.reshape(3, 1)).squeeze(-1)  # (M,3)
    t_new = t_arr + RT                           # (M,3)

    # sanity check: centers shift by -T
    C_new = -np.transpose(R_arr, (0, 2, 1)) @ t_new[..., None]  # (M,3,1)
    C_new = C_new.squeeze(-1)                                   # (M,3)
    if not np.allclose(C_new, C_arr - T, atol=1e-6):
        raise AssertionError("Camera centers not shifted correctly. Check math/conventions.")

    # write back tvec
    for i, img_id in enumerate(img_ids):
        images[img_id].tvec = t_new[i].astype(np.float64)

    # --- Save recentered model ---
    colmap.write_model(cameras, images, points3D, out_sparse_dir, ext=".bin")

    print(f"✅ Correct Roration Center: saved recentered COLMAP model to {out_sparse_dir}")

    return {
        "T": T,
        "used": used,
        "num_images": len(images),
        "num_points": len(points3D),
        "out_dir": out_sparse_dir,
    }

def parse_args():
    parser = argparse.ArgumentParser(
        description="Recenter a COLMAP sparse model (e.g. for Gaussian Splatting input)."
    )
    parser.add_argument(
        "--in_sparse_dir",
        type=str,
        required=True,
        help="Path to input sparse/0 directory containing COLMAP model files.",
    )
    parser.add_argument(
        "--out_sparse_dir",
        type=str,
        required=True,
        help="Path to output sparse/0 directory to save recentered model.",
    )
    parser.add_argument(
        "--center_by",
        type=str,
        default="auto",
        choices=["auto", "points", "cameras"],
        help="Choose centering mode: 'points', 'cameras', or 'auto' (default).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output folder if it already exists.",
    )
    return parser.parse_args()


# ---------- Example usage ----------
if __name__ == "__main__":
    args = parse_args()
    recenter_colmap_model(
        in_sparse_dir=args.in_sparse_dir,
        out_sparse_dir=args.out_sparse_dir,
        center_by=args.center_by,
        overwrite=args.overwrite,
    )
