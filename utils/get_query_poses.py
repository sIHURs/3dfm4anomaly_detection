#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert query pose jsonl (one JSON per line) into a NeRF/3DGS-style transforms.json.

- Reads camera_angle_x from an existing transforms_anomaly_free_poses.json
- Each jsonl line must contain:
    - query_path (str)
    - query_transform_matrix (4x4 list)
- Outputs:
    {
      "camera_angle_x": ...,
      "frames": [
        {"file_path": "...", "transform_matrix": [...]},
        ...
      ]
    }
"""

import argparse
import json
import os
from typing import Any, Dict, List


def read_camera_angle_x(transforms_path: str) -> float:
    with open(transforms_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "camera_angle_x" not in data:
        raise KeyError(f"'camera_angle_x' not found in {transforms_path}")
    return float(data["camera_angle_x"])


def iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield ln, json.loads(line)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Invalid JSON at line {ln} in {path}: {e}") from e


def make_file_path(query_path: str, subset: str | None, mode: str, root: str | None) -> str:
    """
    mode:
      - basename: only filename
      - subset_basename: subset/filename
      - rel_root: relative to root (requires root)
      - keep: keep original absolute path
    """
    if mode == "keep":
        return query_path

    base = os.path.basename(query_path)

    if mode == "basename":
        return base

    if mode == "subset_basename":
        if subset:
            return os.path.join(subset, base).replace("\\", "/")
        return base

    if mode == "rel_root":
        if not root:
            raise ValueError("--root is required when --path_mode=rel_root")
        rel = os.path.relpath(query_path, root)
        return rel.replace("\\", "/")

    raise ValueError(f"Unknown path_mode: {mode}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True, help="Input .jsonl file (one JSON object per line)")
    ap.add_argument(
        "--anomaly_free_transforms",
        required=True,
        help="Path to transforms_anomaly_free_poses.json (source of camera_angle_x)",
    )
    ap.add_argument("--out", required=True, help="Output transforms.json path")
    ap.add_argument(
        "--path_mode",
        default="basename",
        choices=["basename", "subset_basename", "rel_root", "keep"],
        help="How to write frame.file_path",
    )
    ap.add_argument(
        "--root",
        default=None,
        help="Root dir used for --path_mode=rel_root, e.g. dataset root",
    )
    ap.add_argument(
        "--sort_by_transforms_json",
        action="store_true",
        help="Sort frames by the numeric id inside 'transforms_json' filename if present.",
    )
    args = ap.parse_args()

    camera_angle_x = read_camera_angle_x(args.anomaly_free_transforms)

    frames: List[Dict[str, Any]] = []
    records_for_sort: List[tuple[int, Dict[str, Any]]] = []

    for ln, obj in iter_jsonl(args.jsonl):
        if "query_path" not in obj or "query_transform_matrix" not in obj:
            raise KeyError(f"Line {ln}: missing 'query_path' or 'query_transform_matrix'")

        query_path = obj["query_path"]
        subset = obj.get("subset")
        T = obj["query_transform_matrix"]

        if not (isinstance(T, list) and len(T) == 4 and all(isinstance(r, list) and len(r) == 4 for r in T)):
            raise ValueError(f"Line {ln}: 'query_transform_matrix' must be 4x4 list")

        file_path = make_file_path(query_path, subset, args.path_mode, args.root)

        frame = {
            "file_path": file_path,
            "transform_matrix": T,
        }

        # 可选：如果你想在 transforms.json 里保留 subset 信息（非标准字段）
        # frame["subset"] = subset

        if args.sort_by_transforms_json:
            # try parse .../transforms_<subset>_00016_queryX.json -> 16
            tid = -1
            tj = obj.get("transforms_json", "")
            name = os.path.basename(tj)
            # find the first 5-digit block or any digits block after an underscore
            import re
            m = re.search(r"_(\d+)_", name) or re.search(r"_(\d+)\D", name)
            if m:
                try:
                    tid = int(m.group(1))
                except ValueError:
                    tid = -1
            records_for_sort.append((tid, frame))
        else:
            frames.append(frame)

    if args.sort_by_transforms_json:
        # stable sort: tid=-1 will come first; if you prefer last, change key accordingly
        frames = [fr for _, fr in sorted(records_for_sort, key=lambda x: x[0])]

    out_data = {
        "camera_angle_x": camera_angle_x,
        "frames": frames,
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2)

    print(f"[OK] Wrote {len(frames)} frames -> {args.out}")
    print(f"[OK] camera_angle_x = {camera_angle_x} (from {args.anomaly_free_transforms})")


if __name__ == "__main__":
    main()
