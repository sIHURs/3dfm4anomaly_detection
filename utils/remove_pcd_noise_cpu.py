from plyfile import PlyData, PlyElement
import numpy as np
import cv2
import colmap
from tqdm import tqdm
import os
import argparse


def load_colmap_model(colmap_dir):
    """
    Load COLMAP cameras and images from a sparse model directory.
    """
    cameras, images, points3D = colmap.read_model(colmap_dir, ext=".bin")
    return cameras, images


def project_point(X, camera, image):
    """
    Project a 3D point into the image using COLMAP camera parameters.
    Returns (u, v) in pixel coordinates, or None if point is behind the camera.
    """
    R = colmap.qvec2rotmat(image.qvec)
    t = image.tvec.reshape(3, 1)

    X = X.reshape(3, 1)
    X_cam = R @ X + t

    # point behind the camera
    if X_cam[2] <= 1e-6:
        return None

    fx, fy, cx, cy = camera.params[0:4]
    u = fx * (X_cam[0] / X_cam[2]) + cx
    v = fy * (X_cam[1] / X_cam[2]) + cy
    return float(u), float(v)


def detect_ply_type(vertex, fallback="pointcloud"):
    """
    Detect whether a PLY file is a 3D Gaussian Splatting PLY (3DGS)
    or a normal point cloud based on vertex property names.
    """
    names = set(vertex.data.dtype.names or [])

    # Typical 3DGS attributes
    three_dgs_signals = {
        "scale_0", "scale_1", "scale_2",
        "rotation",
        "opacity",
        "f_dc_0", "f_rest_0", "f_rest_1",
        "sh0", "sh1", "sh2"
    }

    if names & three_dgs_signals:
        return "3dgs"

    # Otherwise treat as normal point cloud
    return fallback


def filter_gaussians(
    ply_path,
    sparse_dir,
    image_dir,
    output_path,
    outside_threshold=0.6,
    input_type="auto",
):
    """
    Filter points from a PLY file (3DGS or normal point cloud) using alpha masks.

    - Projects each 3D point into all available RGBA images.
    - Alpha=0 pixels mark "outside" projections.
    - For each point, compute ratio = (# outside) / (# valid projections).
    - Remove points with ratio >= outside_threshold.

    This function keeps all vertex attributes intact and only removes points.
    """

    # Load input PLY
    ply = PlyData.read(ply_path)
    vertex = ply["vertex"]
    N = len(vertex)

    # Determine PLY type
    if input_type == "auto":
        detected = detect_ply_type(vertex)
        print(f"[INFO] Detected PLY type: {detected} (auto)")
        ply_type = detected
    else:
        ply_type = input_type
        print(f"[INFO] Using user-specified PLY type: {ply_type}")

    # Ensure xyz exists
    if not all(name in vertex.data.dtype.names for name in ("x", "y", "z")):
        raise RuntimeError(
            f"PLY does not contain x,y,z properties: {vertex.data.dtype.names}"
        )

    # Extract xyz coordinates
    pts = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1)
    print(f"Loaded {N} points from {ply_type} PLY: {ply_path}")

    # Load COLMAP models
    cameras, images = load_colmap_model(sparse_dir)
    images_list = list(images.values())

    # Load alpha masks from RGBA images
    mask_cache = {}
    for img in images_list:
        img_path = os.path.join(image_dir, img.name)
        rgba = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

        if rgba is None or rgba.shape[2] != 4:
            raise RuntimeError(f"Cannot read RGBA image (4 channels required): {img_path}")

        alpha = rgba[:, :, 3]
        _, mask = cv2.threshold(alpha, 0, 255, cv2.THRESH_BINARY)
        mask_cache[img.id] = mask

    keep = np.zeros(N, dtype=bool)

    print("Filtering points using alpha masks...")
    for i in tqdm(range(N)):
        X = pts[i]

        total = 0     # number of valid projections
        outside = 0   # number of projections landing on alpha=0

        for img in images_list:
            cam = cameras[img.camera_id]
            mask = mask_cache[img.id]

            H, W = mask.shape

            pix = project_point(X, cam, img)
            if pix is None:
                continue

            u, v = int(pix[0]), int(pix[1])
            if u < 0 or u >= W or v < 0 or v >= H:
                continue

            total += 1
            if mask[v, u] == 0:
                outside += 1

        # Keep point if outside ratio is below threshold
        if total > 0:
            ratio = outside / total
            keep[i] = ratio < outside_threshold

    kept_indices = np.where(keep)[0]
    new_vertices = vertex[kept_indices]

    print(f"Remaining: {len(new_vertices)} / {N} points")

    # Recreate new PLY with same structure, but fewer points
    new_ply = PlyData(
        [PlyElement.describe(new_vertices, "vertex")],
        text=ply.text
    )
    new_ply.write(output_path)
    print("Saved:", output_path)


def main():
    parser = argparse.ArgumentParser(description="Filter Gaussian / point cloud PLY using image alpha masks")

    parser.add_argument("--ply_path", type=str, required=True,
                        help="Path to input .ply file (3DGS or normal point cloud)")

    parser.add_argument("--sparse_dir", type=str, required=True,
                        help="COLMAP sparse model directory")

    parser.add_argument("--image_dir", type=str, required=True,
                        help="Directory containing RGBA images")

    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to save filtered PLY file")

    parser.add_argument("--outside_threshold", type=float, default=0.1,
                        help="Max allowed outside-projection ratio per point")

    parser.add_argument(
        "--input_type",
        type=str,
        default="auto",
        choices=["auto", "3dgs", "pointcloud"],
        help=(
            "Type of input PLY:\n"
            "'auto' = auto-detect from PLY fields (default)\n"
            "'3dgs' = force treat as Gaussian Splatting PLY\n"
            "'pointcloud' = force treat as standard point cloud PLY"
        ),
    )

    args = parser.parse_args()

    filter_gaussians(
        ply_path=args.ply_path,
        sparse_dir=args.sparse_dir,
        image_dir=args.image_dir,
        output_path=args.output_path,
        outside_threshold=args.outside_threshold,
        input_type=args.input_type,
    )


if __name__ == "__main__":
    main()

# Test logs example:
# 0.0  => broken?
# 0.03 => 58376 points
# 0.05 => 66036 points
# 0.1  => ...