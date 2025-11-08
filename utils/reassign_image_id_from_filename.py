#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rewrite COLMAP image IDs (and camera IDs) in a sparse model folder (sparse/0)
based on filename:
    new_id = <number extracted from IMAGE_NAME> + 1
Then propagate:
- images.txt: IMAGE_ID -> new_id, CAMERA_ID -> new_id
- cameras.txt: duplicate/clone per-image camera with CAMERA_ID=new_id
- points3D.txt: remap tracks' IMAGE_ID -> new_id

Supports TXT or BIN input (uses `colmap model_converter` if only BIN exists).
Optionally converts edited TXT back to BIN.

Example:
    python remap_sparse_ids.py --sparse path/to/sparse/0 --out path/to/output_txt --to-bin
"""

import argparse
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Tuple, List

# -----------------------
# Utilities
# -----------------------

def has_txt_model(p: Path) -> bool:
    return (p/"cameras.txt").exists() and (p/"images.txt").exists() and (p/"points3D.txt").exists()

def has_bin_model(p: Path) -> bool:
    return (p/"cameras.bin").exists() and (p/"images.bin").exists() and (p/"points3D.bin").exists()

def run(cmd: List[str]):
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
    return proc.stdout

def ensure_txt_from_sparse(sparse_dir: Path, colmap_exe: str) -> Path:
    if has_txt_model(sparse_dir):
        return sparse_dir
    if has_bin_model(sparse_dir):
        tmpdir = Path(tempfile.mkdtemp(prefix="colmap_txt_"))
        run([colmap_exe, "model_converter",
             "--input_path", str(sparse_dir),
             "--output_path", str(tmpdir),
             "--output_type", "TXT"])
        return tmpdir
    raise FileNotFoundError("No COLMAP model found (neither TXT nor BIN). Expected cameras/images/points3D in TXT or BIN.")

# -----------------------
# Parse / Write COLMAP TXT
# -----------------------

def parse_cameras_txt(path: Path) -> Dict[int, Tuple]:
    """
    cameras.txt format:
      CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
    Return dict:
      cam_id -> (model, width, height, params_list[str])
    """
    cams = {}
    with open(path, "r") as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            toks = ln.split()
            cam_id = int(toks[0])
            model = toks[1]
            width = int(toks[2])
            height = int(toks[3])
            params = toks[4:]
            cams[cam_id] = (model, width, height, params)
    return cams

def write_cameras_txt(path: Path, cams: Dict[int, Tuple]):
    with open(path, "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        for cam_id, (model, width, height, params) in cams.items():
            f.write(f"{cam_id} {model} {width} {height} {' '.join(map(str, params))}\n")

def parse_images_txt(path: Path) -> Tuple[Dict[int, Tuple], Dict[int, str], Dict[str, int], List[int]]:
    images = {}
    id2name, name2id, order = {}, {}, []
    with open(path, "r") as f:
        lines = [ln.rstrip("\n") for ln in f]

    i = 0
    while i < len(lines):
        ln = lines[i].strip()
        if not ln or ln.startswith("#"):
            i += 1
            continue
        header = ln
        pts2d = ""
        if i + 1 < len(lines):
            pts2d = lines[i+1].strip()
        i += 2

        toks = header.split()
        img_id = int(toks[0])
        qw, qx, qy, qz = map(float, toks[1:5])
        tx, ty, tz = map(float, toks[5:8])
        cam_id = int(toks[8])
        name = " ".join(toks[9:])

        images[img_id] = (qw, qx, qy, qz, tx, ty, tz, cam_id, name, pts2d)
        id2name[img_id] = name
        name2id[name] = img_id
        order.append(img_id)
    return images, id2name, name2id, order

def write_images_txt(path: Path, images: Dict[int, Tuple], order: List[int]):
    with open(path, "w") as f:
        f.write("# Image list with two lines per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, IMAGE_NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for img_id in order:
            qw,qx,qy,qz,tx,ty,tz,cam_id,name,pts2d = images[img_id]
            f.write(f"{img_id} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {cam_id} {name}\n")
            f.write(f"{pts2d}\n")

def parse_points3d_txt(path: Path) -> Dict[int, Tuple]:
    pts = {}
    with open(path, "r") as f:
        for ln in f:
            if not ln.strip() or ln.startswith("#"):
                continue
            toks = ln.strip().split()
            pid = int(toks[0])
            X, Y, Z = map(float, toks[1:4])
            R, G, B = map(int, toks[4:7])
            ERR = float(toks[7])
            rest = toks[8:]
            track = []
            for a, b in zip(rest[0::2], rest[1::2]):
                track.append([int(a), int(b)])  # (image_id, point2d_idx)
            pts[pid] = (X, Y, Z, R, G, B, ERR, track)
    return pts

def write_points3d_txt(path: Path, pts: Dict[int, Tuple]):
    with open(path, "w") as f:
        f.write("# 3D point list with one line per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        for pid, (X, Y, Z, R, G, B, ERR, track) in pts.items():
            flat = " ".join(f"{iid} {idx}" for iid, idx in track)
            f.write(f"{pid} {X} {Y} {Z} {R} {G} {B} {ERR} {flat}\n")

# -----------------------
# ID Remapping Rule
# -----------------------

_num_pat = re.compile(r"(\d+)")

def new_id_from_name_plus1(name: str) -> int:
    m = _num_pat.search(name)
    if not m:
        raise ValueError(f"No integer found in image name: {name}")
    return int(m.group(1)) + 1

def build_id_mapping_by_filename(id2name: Dict[int, str]) -> Dict[int, int]:
    old2new = {}
    used = set()
    for old_id, name in id2name.items():
        new_id = new_id_from_name_plus1(name)
        if new_id in used:
            raise ValueError(f"New image_id collision: {new_id} for name {name}. Check your filenames.")
        used.add(new_id)
        old2new[old_id] = new_id
    return old2new

# -----------------------
# Remap helpers
# -----------------------

def remap_images_and_cameras_per_image(
    images: Dict[int, Tuple],
    cameras: Dict[int, Tuple],
    order: List[int],
    old2new_img: Dict[int, int],
) -> Tuple[Dict[int, Tuple], List[int], Dict[int, Tuple]]:
    """
    - images_new: use new IMAGE_ID; set CAMERA_ID=new IMAGE_ID
    - cameras_new: for each image, clone the old camera params into CAMERA_ID=new IMAGE_ID
    """
    images_new = {}
    order_new = []
    cameras_new = {}

    for old_img_id in order:
        new_img_id = old2new_img[old_img_id]
        qw,qx,qy,qz,tx,ty,tz,old_cam_id,name,pts2d = images[old_img_id]

        # 1) clone camera params to new CAMERA_ID=new_img_id
        if old_cam_id not in cameras:
            raise KeyError(f"Camera {old_cam_id} (from image {old_img_id}) not found in cameras.txt")
        cam_model, w, h, params = cameras[old_cam_id]
        cameras_new[new_img_id] = (cam_model, w, h, params)

        # 2) write image with CAMERA_ID=new_img_id
        images_new[new_img_id] = (qw,qx,qy,qz,tx,ty,tz,new_img_id,name,pts2d)
        order_new.append(new_img_id)

    return images_new, order_new, cameras_new

def remap_points_tracks(pts: Dict[int, Tuple], old2new_img: Dict[int, int]) -> Dict[int, Tuple]:
    new_pts = {}
    for pid, (X, Y, Z, R, G, B, ERR, track) in pts.items():
        track2 = []
        for iid, idx in track:
            if iid not in old2new_img:
                raise KeyError(f"image_id {iid} in points3D not found in images.txt mapping.")
            track2.append([old2new_img[iid], idx])
        new_pts[pid] = (X, Y, Z, R, G, B, ERR, track2)
    return new_pts

# -----------------------
# Main
# -----------------------

def main():
    ap = argparse.ArgumentParser(
        description="Rewrite COLMAP IMAGE_ID and CAMERA_ID (both = filename-number+1), and update points3D tracks."
    )
    ap.add_argument("--sparse", required=True, type=Path, help="Path to COLMAP sparse model directory (e.g., .../sparse/0).")
    ap.add_argument("--out", required=True, type=Path, help="Output directory for the edited TXT model.")
    ap.add_argument("--to-bin", action="store_true", help="Also convert the edited TXT model to BIN (requires `colmap` in PATH).")
    ap.add_argument("--colmap", default="colmap", help="Path to the COLMAP executable (default: colmap).")
    args = ap.parse_args()

    sparse_dir = args.sparse.resolve()
    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Ensure TXT model
    txt_dir = ensure_txt_from_sparse(sparse_dir, args.colmap)

    # 2) Read TXT model
    cameras_src = txt_dir / "cameras.txt"
    images_src  = txt_dir / "images.txt"
    points_src  = txt_dir / "points3D.txt"

    cameras = parse_cameras_txt(cameras_src)
    images, id2name, name2id, order = parse_images_txt(images_src)
    pts = parse_points3d_txt(points_src)

    # 3) Mapping old_image_id -> new_image_id (= filename number + 1)
    old2new_img = build_id_mapping_by_filename(id2name)

    # 4) Apply mapping
    images_new, order_new, cameras_new = remap_images_and_cameras_per_image(images, cameras, order, old2new_img)
    pts_new = remap_points_tracks(pts, old2new_img)

    # 5) Write outputs (TXT)
    write_images_txt(out_dir / "images.txt", images_new, order_new)
    write_cameras_txt(out_dir / "cameras.txt", cameras_new)
    write_points3d_txt(out_dir / "points3D.txt", pts_new)

    print(f"[OK] Wrote edited TXT model to: {out_dir}")

    # 6) Optional BIN conversion
    if args.to_bin:
        bin_dir = out_dir.parent / (out_dir.name + "_bin")
        bin_dir.mkdir(parents=True, exist_ok=True)
        run([args.colmap, "model_converter",
             "--input_path", str(out_dir),
             "--output_path", str(bin_dir),
             "--output_type", "BIN"])
        print(f"[OK] Also wrote BIN model to: {bin_dir}")

if __name__ == "__main__":
    main()
