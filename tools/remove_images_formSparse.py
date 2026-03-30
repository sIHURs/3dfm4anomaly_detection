#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Set


def build_remove_names() -> Set[str]:
    names = set()
    for i in range(0, 15):
        names.add(f"train_{i:03d}.png")
    for i in range(195, 210):
        names.add(f"train_{i:03d}.png")
    return names


def read_images_txt(images_path: Path):
    """
    Parse COLMAP images.txt

    Returns:
        header_lines: comment/header lines
        images: list of dict:
            {
                "image_id": int,
                "meta_line": str,
                "points2d_line": str,
                "name": str,
            }
    """
    with images_path.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    header_lines = []
    data_lines = []

    for line in lines:
        if line.startswith("#"):
            header_lines.append(line)
        elif line.strip():
            data_lines.append(line.rstrip("\n"))

    if len(data_lines) % 2 != 0:
        raise ValueError("images.txt 格式异常：非注释有效行数不是偶数。")

    images = []
    for i in range(0, len(data_lines), 2):
        meta_line = data_lines[i]
        points2d_line = data_lines[i + 1]

        parts = meta_line.split()
        if len(parts) < 10:
            raise ValueError(f"images.txt meta line 格式异常：\n{meta_line}")

        image_id = int(parts[0])
        name = parts[-1]

        images.append({
            "image_id": image_id,
            "meta_line": meta_line,
            "points2d_line": points2d_line,
            "name": name,
        })

    return header_lines, images


def write_images_txt(out_path: Path, header_lines: List[str], images: List[dict]):
    with out_path.open("w", encoding="utf-8") as f:
        for line in header_lines:
            f.write(line if line.endswith("\n") else line + "\n")
        for img in images:
            f.write(img["meta_line"].rstrip("\n") + "\n")
            f.write(img["points2d_line"].rstrip("\n") + "\n")


def read_points3d_txt(points_path: Path):
    """
    Parse COLMAP points3D.txt

    Each valid line:
    POINT3D_ID X Y Z R G B ERROR TRACK[] as (IMAGE_ID, POINT2D_IDX) pairs
    """
    with points_path.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    header_lines = []
    point_lines = []

    for line in lines:
        if line.startswith("#"):
            header_lines.append(line)
        elif line.strip():
            point_lines.append(line.rstrip("\n"))

    return header_lines, point_lines


def filter_points3d_lines(point_lines: List[str], removed_image_ids: Set[int]) -> List[str]:
    """
    Remove track entries referencing removed images.
    If a point has <2 observations after filtering, drop it.
    """
    kept_lines = []

    for line in point_lines:
        parts = line.split()
        if len(parts) < 8:
            continue

        fixed = parts[:8]
        track = parts[8:]

        if len(track) % 2 != 0:
            raise ValueError(f"points3D.txt track 格式异常：\n{line}")

        new_track = []
        for i in range(0, len(track), 2):
            image_id = int(track[i])
            point2d_idx = track[i + 1]
            if image_id not in removed_image_ids:
                new_track.extend([str(image_id), point2d_idx])

        # 一个 3D 点通常至少要有 2 个观测才有效
        if len(new_track) < 4:
            continue

        kept_lines.append(" ".join(fixed + new_track))

    return kept_lines


def write_points3d_txt(out_path: Path, header_lines: List[str], point_lines: List[str]):
    with out_path.open("w", encoding="utf-8") as f:
        for line in header_lines:
            f.write(line if line.endswith("\n") else line + "\n")
        for line in point_lines:
            f.write(line.rstrip("\n") + "\n")


def copy_cameras_txt(src: Path, dst: Path):
    dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="从 COLMAP text model 中删除指定图片，并同步更新 points3D tracks。"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="输入 COLMAP text model 文件夹，包含 cameras.txt / images.txt / points3D.txt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="输出文件夹",
    )
    args = parser.parse_args()

    input_dir: Path = args.input
    output_dir: Path = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    cameras_in = input_dir / "cameras.txt"
    images_in = input_dir / "images.txt"
    points_in = input_dir / "points3D.txt"

    cameras_out = output_dir / "cameras.txt"
    images_out = output_dir / "images.txt"
    points_out = output_dir / "points3D.txt"

    for p in [cameras_in, images_in, points_in]:
        if not p.exists():
            raise FileNotFoundError(f"缺少文件: {p}")

    remove_names = build_remove_names()

    # --- images.txt ---
    header_lines, images = read_images_txt(images_in)

    kept_images = []
    removed_image_ids = set()

    for img in images:
        if Path(img["name"]).name in remove_names:
            removed_image_ids.add(img["image_id"])
        else:
            kept_images.append(img)

    write_images_txt(images_out, header_lines, kept_images)

    # --- points3D.txt ---
    p_header, p_lines = read_points3d_txt(points_in)
    new_p_lines = filter_points3d_lines(p_lines, removed_image_ids)
    write_points3d_txt(points_out, p_header, new_p_lines)

    # --- cameras.txt ---
    copy_cameras_txt(cameras_in, cameras_out)

    print("处理完成")
    print(f"输入目录:  {input_dir}")
    print(f"输出目录:  {output_dir}")
    print(f"删除图片数: {len(removed_image_ids)}")
    print("删除的图片名:")
    for n in sorted(remove_names):
        print("  ", n)


if __name__ == "__main__":
    main()