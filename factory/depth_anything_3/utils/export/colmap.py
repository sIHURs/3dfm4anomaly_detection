# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import pycolmap
import cv2 as cv
import numpy as np

from PIL import Image

from depth_anything_3.specs import Prediction
from depth_anything_3.utils.logger import logger

from .glb import _depths_to_world_points_with_colors


def _compute_resize_pad(orig_w, orig_h, proc_w, proc_h, method: str):
    """
    Return (scale, pad_x, pad_y) such that:
      processed_x = orig_x * scale + pad_x
      processed_y = orig_y * scale + pad_y

    We assume processed_images are produced by:
      - 'resize' (direct resize): scale_x = proc_w/orig_w, scale_y = proc_h/orig_h, no pad
      - 'upper_bound_resize' (keep aspect + pad to proc size): scale = min(proc_w/orig_w, proc_h/orig_h), pad centered
    """
    if method.endswith("resize") and method not in ("upper_bound_resize", "lower_bound_resize"):
        # plain resize (non keep-aspect)
        sx = proc_w / float(orig_w)
        sy = proc_h / float(orig_h)
        return (sx, 0.0, 0.0, sy, 0.0, 0.0)  # (sx, px, py, sy, px2, py2) keep format consistent

    if method == "upper_bound_resize":
        scale = min(proc_w / float(orig_w), proc_h / float(orig_h))
        new_w = orig_w * scale
        new_h = orig_h * scale
        pad_x = (proc_w - new_w) * 0.5
        pad_y = (proc_h - new_h) * 0.5
        return (scale, pad_x, pad_y, scale, pad_x, pad_y)

    if method == "lower_bound_resize":
        # common keep-aspect then center-crop to proc size
        # Here inverse mapping needs crop offsets; without explicit metadata, we approximate as centered crop.
        scale = max(proc_w / float(orig_w), proc_h / float(orig_h))
        new_w = orig_w * scale
        new_h = orig_h * scale
        crop_x = (new_w - proc_w) * 0.5
        crop_y = (new_h - proc_h) * 0.5
        # processed = orig*scale - crop_offset
        # => orig = (processed + crop_offset)/scale
        return (scale, -crop_x, -crop_y, scale, -crop_x, -crop_y)

    if method == "crop":
        raise NotImplementedError("COLMAP export for crop method is not implemented")

    # fallback: treat as direct resize
    sx = proc_w / float(orig_w)
    sy = proc_h / float(orig_h)
    return (sx, 0.0, 0.0, sy, 0.0, 0.0)


def _map_processed_xy_to_orig_xy(xy_proc, orig_w, orig_h, proc_w, proc_h, method: str):
    """
    xy_proc: (2,) in processed image coords
    Return xy_orig in original image coords.
    """
    sx, px, py, sy, _, _ = _compute_resize_pad(orig_w, orig_h, proc_w, proc_h, method)
    x_p, y_p = float(xy_proc[0]), float(xy_proc[1])

    # Inverse mapping:
    # processed = orig * scale + pad  => orig = (processed - pad)/scale
    x_o = (x_p - px) / sx if sx != 0 else x_p
    y_o = (y_p - py) / sy if sy != 0 else y_p
    return np.array([x_o, y_o], dtype=np.float64)


def _map_processed_intri_to_orig_intri(intri_proc, orig_w, orig_h, proc_w, proc_h, method: str):
    """
    intri_proc: (3,3) for processed images.
    Return intri_orig: (3,3) for original images.

    For keep-aspect+pad:
      x_proc = x_orig*scale + pad_x
      => cx_proc = cx_orig*scale + pad_x  => cx_orig = (cx_proc - pad_x)/scale
      => fx_proc = fx_orig*scale          => fx_orig = fx_proc/scale
    """
    intri = intri_proc.copy().astype(np.float64)

    sx, px, py, sy, _, _ = _compute_resize_pad(orig_w, orig_h, proc_w, proc_h, method)

    # fx, fy
    if sx != 0:
        intri[0, 0] = intri[0, 0] / sx
    if sy != 0:
        intri[1, 1] = intri[1, 1] / sy

    # cx, cy
    if sx != 0:
        intri[0, 2] = (intri[0, 2] - px) / sx
    if sy != 0:
        intri[1, 2] = (intri[1, 2] - py) / sy

    return intri


def _select_topk_points_by_conf(conf: np.ndarray, k: int):
    """
    conf: (F,H,W) float
    Return flat indices (int64) for top-k confidence pixels.
    """
    flat = conf.reshape(-1)
    if k <= 0:
        return np.empty((0,), dtype=np.int64)
    k = min(k, flat.size)

    # argpartition for speed, then sort those k
    idx_part = np.argpartition(-flat, kth=k - 1)[:k]
    idx_sorted = idx_part[np.argsort(-flat[idx_part])]
    return idx_sorted.astype(np.int64)


def export_to_colmap(
    prediction: Prediction,
    export_dir: str,
    image_paths: list[str],
    conf_thresh_percentile: float = 40.0,
    process_res_method: str = "upper_bound_resize",
    num_points: int | None = None,
    point_select: str = "topk",
    seed: int = 0,
) -> None:

    # processed resolution
    num_frames = len(prediction.processed_images)
    proc_h, proc_w = prediction.processed_images.shape[1:3]

    if num_points is not None:
        if point_select == "topk":
            flat_idx = _select_topk_points_by_conf(prediction.conf, num_points)
        elif point_select == "random":
            rng = np.random.default_rng(seed)
            flat = prediction.conf.reshape(-1)
            # 按 conf 做加权采样（也可以改成均匀）
            prob = flat / (flat.sum() + 1e-12)
            k = min(num_points, flat.size)
            flat_idx = rng.choice(flat.size, size=k, replace=False, p=prob).astype(np.int64)
        else:
            raise ValueError(f"Unknown point_select: {point_select}")
        
        flat_conf = prediction.conf.reshape(-1)
        kth_conf = float(flat_conf[flat_idx[-1]]) if flat_idx.size > 0 else 1.0
        conf_thresh = kth_conf
        logger.info(f"[DA3->COLMAP] Using topK conf threshold ~ {conf_thresh:.6f} to target {num_points} points")
    else:
        conf_thresh = float(np.percentile(prediction.conf, conf_thresh_percentile))
        logger.info(f"[DA3->COLMAP] Using percentile conf threshold {conf_thresh_percentile}% => {conf_thresh:.6f}")

    points, colors = _depths_to_world_points_with_colors(
        prediction.depth,
        prediction.intrinsics,
        prediction.extrinsics,  # w2c
        prediction.processed_images,
        prediction.conf,
        conf_thresh,
    )

    points = np.asarray(points)
    colors = np.asarray(colors)

    points_xyf = _create_xyf(num_frames, proc_h, proc_w)
    mask = (prediction.conf >= conf_thresh)
    points_xyf = points_xyf[mask.reshape(-1)]

    if num_points is not None:
        conf_kept = prediction.conf[mask].reshape(-1)
        n_kept = conf_kept.size

        if n_kept == 0:
            logger.warning("[DA3->COLMAP] No points after thresholding; exporting empty reconstruction.")
        else:
            k = min(num_points, n_kept)
            order = np.argsort(-conf_kept)[:k]

            points = points[order]
            colors = colors[order]
            points_xyf = points_xyf[order]

            if k < num_points:
                logger.warning(
                    f"[DA3->COLMAP] Only {k} points available (<{num_points}). "
                    "Exporting fewer points."
                )

    num_points_final = int(len(points))
    logger.info(f"[DA3->COLMAP] Exporting to COLMAP with {num_points_final} points")

    reconstruction = pycolmap.Reconstruction()

    point3d_ids = []
    for vidx in range(num_points_final):
        point3d_id = reconstruction.add_point3D(points[vidx], pycolmap.Track(), colors[vidx])
        point3d_ids.append(point3d_id)

    for fidx in range(num_frames):
        orig_w, orig_h = Image.open(image_paths[fidx]).size

        intri_proc = prediction.intrinsics[fidx]  # processed intrinsics
        intri_orig = _map_processed_intri_to_orig_intri(
            intri_proc, orig_w, orig_h, proc_w, proc_h, process_res_method
        )

        pycolmap_intri = np.array(
            [intri_orig[0, 0], intri_orig[1, 1], intri_orig[0, 2], intri_orig[1, 2]],
            dtype=np.float64
        )

        extrinsic = prediction.extrinsics[fidx]  # w2c
        cam_from_world = pycolmap.Rigid3d(
            pycolmap.Rotation3d(extrinsic[:3, :3]),
            extrinsic[:3, 3]
        )

        # camera
        camera = pycolmap.Camera()
        camera.camera_id = fidx + 1
        camera.model = pycolmap.CameraModelId.PINHOLE
        camera.width = orig_w
        camera.height = orig_h
        camera.params = pycolmap_intri
        reconstruction.add_camera(camera)

        # rig
        rig = pycolmap.Rig()
        rig.rig_id = camera.camera_id
        rig.add_ref_sensor(camera.sensor_id)
        reconstruction.add_rig(rig)

        # image
        image = pycolmap.Image()
        image.image_id = fidx + 1
        image.camera_id = camera.camera_id

        # frame
        frame = pycolmap.Frame()
        frame.frame_id = image.image_id
        frame.rig_id = camera.camera_id
        frame.add_data_id(image.data_id)
        frame.rig_from_world = cam_from_world
        reconstruction.add_frame(frame)

        # points2D + track
        point2d_list = []
        points_in_frame = points_xyf[:, 2].astype(np.int32) == fidx
        idxs = np.where(points_in_frame)[0]

        for local_i, vidx in enumerate(idxs):
            xy_proc = points_xyf[vidx][:2].astype(np.float64)

            xy_orig = _map_processed_xy_to_orig_xy(
                xy_proc, orig_w, orig_h, proc_w, proc_h, process_res_method
            )

            point3d_id = point3d_ids[vidx]
            point2d_list.append(pycolmap.Point2D(xy_orig, point3d_id))

            reconstruction.point3D(point3d_id).track.add_element(
                image.image_id, len(point2d_list) - 1
            )

        image.frame_id = image.image_id
        image.name = os.path.basename(image_paths[fidx])
        image.points2D = pycolmap.Point2DList(point2d_list)
        reconstruction.add_image(image)

    reconstruction.write(export_dir)


def _create_xyf(num_frames, height, width):
    """
    Creates a grid of pixel coordinates and frame indices (fidx) for all frames.
    """
    y_grid, x_grid = np.indices((height, width), dtype=np.int32)
    x_grid = x_grid[np.newaxis, :, :]
    y_grid = y_grid[np.newaxis, :, :]

    x_coords = np.broadcast_to(x_grid, (num_frames, height, width))
    y_coords = np.broadcast_to(y_grid, (num_frames, height, width))

    f_idx = np.arange(num_frames, dtype=np.int32)[:, np.newaxis, np.newaxis]
    f_coords = np.broadcast_to(f_idx, (num_frames, height, width))

    points_xyf = np.stack((x_coords, y_coords, f_coords), axis=-1)
    return points_xyf.reshape(-1, 3)