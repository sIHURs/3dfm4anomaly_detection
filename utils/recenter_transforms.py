#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import argparse
import os
import numpy as np
from typing import Literal, Tuple, Dict, Any


def load_T(path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Load T from .json or .npy
    """
    ext = os.path.splitext(path)[1].lower()

    if ext == ".npy":
        T = np.load(path).astype(np.float64).reshape(3,)
        return T, {}
    elif ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        T = np.array(meta["T"], dtype=np.float64).reshape(3,)
        return T, meta
    else:
        raise ValueError("T file must be .json or .npy")


def apply_T_to_transforms(
    transforms_in: str,
    transforms_out: str,
    T: np.ndarray,
    matrix_type: Literal["c2w", "w2c"] = "c2w",
) -> None:
    """
    Apply recenter translation T to transforms.json.

    matrix_type:
      - 'c2w': camera-to-world (NeRF style)
      - 'w2c': world-to-camera
    """

    with open(transforms_in, "r", encoding="utf-8") as f:
        data = json.load(f)

    frames = data.get("frames", [])
    if not frames:
        raise RuntimeError("No frames found in transforms.json")

    for frame in frames:
        M = np.array(frame["transform_matrix"], dtype=np.float64)
        if M.shape != (4, 4):
            raise ValueError("transform_matrix must be 4x4")

        if matrix_type == "c2w":
            # camera center shifts by -T
            M[:3, 3] -= T
        else:
            # world->camera: t' = t + R*T
            R = M[:3, :3]
            t = M[:3, 3]
            M[:3, 3] = t + R @ T

        frame["transform_matrix"] = M.tolist()

    os.makedirs(os.path.dirname(transforms_out), exist_ok=True)
    with open(transforms_out, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved transformed transforms.json to: {transforms_out}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Apply saved recenter translation T to transforms.json"
    )
    p.add_argument("--T_path", type=str, required=True,
                   help="Path to saved T (.json or .npy)")
    p.add_argument("--transforms_in", type=str, required=True,
                   help="Input transforms.json")
    p.add_argument("--transforms_out", type=str, required=True,
                   help="Output transforms.json")
    p.add_argument("--matrix_type", type=str, default="c2w",
                   choices=["c2w", "w2c"],
                   help="Type of transform_matrix (default: c2w)")
    return p.parse_args()


def main():
    args = parse_args()

    T, meta = load_T(args.T_path)
    print(f"Loaded T = {T.tolist()}")

    apply_T_to_transforms(
        transforms_in=args.transforms_in,
        transforms_out=args.transforms_out,
        T=T,
        matrix_type=args.matrix_type,
    )


if __name__ == "__main__":
    main()
