from plyfile import PlyData, PlyElement
import numpy as np
import cv2
import colmap
from tqdm import tqdm
import os
import cv2
import matplotlib.pyplot as plt
import argparse

def load_colmap_model(colmap_dir):
    cameras, images, points3D = colmap.read_model(colmap_dir, ext=".bin")
    return cameras, images


def project_point(X, camera, image):
    R = colmap.qvec2rotmat(image.qvec)
    t = image.tvec.reshape(3, 1)

    X = X.reshape(3, 1)
    X_cam = R @ X + t

    if X_cam[2] <= 1e-6:
        return None

    fx, fy, cx, cy = camera.params[0:4]
    u = fx * (X_cam[0] / X_cam[2]) + cx
    v = fy * (X_cam[1] / X_cam[2]) + cy
    return float(u), float(v)


def filter_gaussians(ply_path, sparse_dir, image_dir, output_path,
                     outside_threshold=0.6):

    # Read full Gaussian PLY
    ply = PlyData.read(ply_path)
    vertex = ply['vertex']
    N = len(vertex)

    # Extract xyz
    pts = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=1)

    print("Loaded {} Gaussians".format(N))

    # Load COLMAP cameras & images
    cameras, images = load_colmap_model(sparse_dir)
    images_list = list(images.values())

    # Load alpha masks from images
    mask_cache = {}
    for img in images_list:
        img_path = os.path.join(image_dir, img.name)
        rgba = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if rgba is None or rgba.shape[2] != 4:
            raise RuntimeError(f"Cannot read RGBA image: {img_path}")

        alpha = rgba[:, :, 3]
        _, mask = cv2.threshold(alpha, 0, 255, cv2.THRESH_BINARY)
        mask_cache[img.id] = mask
        
    keep = np.zeros(N, dtype=bool)

    print("Filtering ghosting Gaussians...")
    for i in tqdm(range(N)):
        X = pts[i]

        total = 0
        outside = 0

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

        if total > 0:
            ratio = outside / total
            keep[i] = ratio < outside_threshold

        # break

    # Filter vertices by boolean mask
    new_vertices = vertex[np.where(keep)[0]]
    print("Remaining:", len(new_vertices))

    new_ply = PlyData(
        [PlyElement.describe(new_vertices, 'vertex')],
        text=ply.text
    )
    new_ply.write(output_path)
    print("Saved:", output_path)

def main():
    parser = argparse.ArgumentParser(description="Filter Gaussians in point cloud")

    parser.add_argument("--ply_path", type=str, required=True,
                        help="Path to input .ply file")

    parser.add_argument("--sparse_dir", type=str, required=True,
                        help="COLMAP sparse directory")

    parser.add_argument("--image_dir", type=str, required=True,
                        help="Directory of images")

    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to save filtered output .ply")

    parser.add_argument("--outside_threshold", type=float, default=0.1,
                        help="Threshold for removing outside Gaussians")

    args = parser.parse_args()

    # Call your function
    filter_gaussians(
        ply_path=args.ply_path,
        sparse_dir=args.sparse_dir,
        image_dir=args.image_dir,
        output_path=args.output_path,
        outside_threshold=args.outside_threshold
    )


if __name__ == "__main__":
    main()

# 0.0 broken?
# 0.03 58376 points
# 0.05 66036
# 0.1