#!/usr/bin/env python3
from pathlib import Path
import argparse

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Overlay an anomaly map onto an original image."
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the original image.",
    )
    parser.add_argument(
        "--anomaly_map",
        type=str,
        required=True,
        help="Path to the anomaly map image.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save the overlay result.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.45,
        help="Blend ratio of the heatmap. Default: 0.45",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=(
            "Optional threshold in [0,1]. "
            "Only pixels with anomaly score > threshold will be overlaid."
        ),
    )
    parser.add_argument(
        "--colormap",
        type=str,
        default="turbo",
        choices=["jet", "turbo", "inferno", "hot", "viridis"],
        help="Colormap for anomaly visualization. Default: turbo",
    )
    parser.add_argument(
        "--save_heatmap",
        type=str,
        default=None,
        help="Optional path to save the resized colored heatmap only.",
    )
    parser.add_argument(
        "--save_gray",
        type=str,
        default=None,
        help="Optional path to save the resized normalized anomaly map in grayscale.",
    )
    return parser.parse_args()


def get_colormap(colormap_name: str) -> int:
    mapping = {
        "jet": cv2.COLORMAP_JET,
        "turbo": cv2.COLORMAP_TURBO,
        "inferno": cv2.COLORMAP_INFERNO,
        "hot": cv2.COLORMAP_HOT,
        # OpenCV has no built-in VIRIDIS in some versions, so fall back if unavailable
        "viridis": getattr(cv2, "COLORMAP_VIRIDIS", cv2.COLORMAP_TURBO),
    }
    return mapping[colormap_name]


def load_image_bgr(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def load_anomaly_map(path: str) -> np.ndarray:
    anomaly = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if anomaly is None:
        raise FileNotFoundError(f"Could not read anomaly map: {path}")

    if anomaly.ndim == 3:
        anomaly = cv2.cvtColor(anomaly, cv2.COLOR_BGR2GRAY)

    return anomaly


def normalize_to_uint8(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    arr = arr.astype(np.float32)
    min_val = float(arr.min())
    max_val = float(arr.max())

    if max_val > min_val:
        norm = (arr - min_val) / (max_val - min_val)
    else:
        norm = np.zeros_like(arr, dtype=np.float32)

    arr_uint8 = (norm * 255.0).clip(0, 255).astype(np.uint8)
    return norm, arr_uint8


def overlay_anomaly_map(
    image_bgr: np.ndarray,
    anomaly_map: np.ndarray,
    alpha: float,
    colormap_name: str,
    threshold: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = image_bgr.shape[:2]

    anomaly_resized = cv2.resize(
        anomaly_map,
        (w, h),
        interpolation=cv2.INTER_LINEAR,
    )

    anomaly_norm, anomaly_uint8 = normalize_to_uint8(anomaly_resized)

    heatmap = cv2.applyColorMap(anomaly_uint8, get_colormap(colormap_name))

    if threshold is None:
        overlay = cv2.addWeighted(image_bgr, 1.0 - alpha, heatmap, alpha, 0)
    else:
        if not (0.0 <= threshold <= 1.0):
            raise ValueError("--threshold must be in [0, 1].")

        overlay = image_bgr.copy()
        blended = cv2.addWeighted(image_bgr, 1.0 - alpha, heatmap, alpha, 0)
        mask = anomaly_norm > threshold
        overlay[mask] = blended[mask]

    return overlay, heatmap, anomaly_uint8


def main():
    args = parse_args()

    image_path = Path(args.image)
    anomaly_map_path = Path(args.anomaly_map)
    output_path = Path(args.output)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.save_heatmap is not None:
        Path(args.save_heatmap).parent.mkdir(parents=True, exist_ok=True)

    if args.save_gray is not None:
        Path(args.save_gray).parent.mkdir(parents=True, exist_ok=True)

    image_bgr = load_image_bgr(str(image_path))
    anomaly_map = load_anomaly_map(str(anomaly_map_path))

    overlay, heatmap, anomaly_gray = overlay_anomaly_map(
        image_bgr=image_bgr,
        anomaly_map=anomaly_map,
        alpha=args.alpha,
        colormap_name=args.colormap,
        threshold=args.threshold,
    )

    ok = cv2.imwrite(str(output_path), overlay)
    if not ok:
        raise RuntimeError(f"Failed to save overlay image to: {output_path}")

    if args.save_heatmap is not None:
        ok = cv2.imwrite(args.save_heatmap, heatmap)
        if not ok:
            raise RuntimeError(f"Failed to save heatmap image to: {args.save_heatmap}")

    if args.save_gray is not None:
        ok = cv2.imwrite(args.save_gray, anomaly_gray)
        if not ok:
            raise RuntimeError(f"Failed to save grayscale anomaly map to: {args.save_gray}")

    print(f"Saved overlay to: {output_path}")
    if args.save_heatmap is not None:
        print(f"Saved heatmap to: {args.save_heatmap}")
    if args.save_gray is not None:
        print(f"Saved normalized grayscale map to: {args.save_gray}")


if __name__ == "__main__":
    main()