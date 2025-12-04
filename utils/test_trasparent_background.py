#!/usr/bin/env python3
import os
import argparse
import cv2
import numpy as np

def process_image(input_path: str, output_path: str, bg_thresh: int = 5):
    """
    Remove black background and make it transparent (RGBA PNG).
    This method detects true black background using RGB channel thresholds
    instead of grayscale, to avoid removing dark-colored objects.

    :param input_path: Path to input image
    :param output_path: Path to output PNG image
    :param bg_thresh: Threshold for detecting black background.
                      A pixel is considered background if R,G,B < bg_thresh.
    """
    img = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"[WARN] Failed to read image: {input_path}")
        return

    # Split channels
    b, g, r = cv2.split(img)

    # Background detection using RGB threshold
    # Only pixels where ALL channels are small are considered true black.
    # This prevents dark objects from being removed.
    mask = (b < bg_thresh) & (g < bg_thresh) & (r < bg_thresh)

    # Generate alpha channel: 0 for background, 255 for foreground
    alpha = np.where(mask, 0, 255).astype(np.uint8)

    # Merge into RGBA image
    rgba = cv2.merge((b, g, r, alpha))

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save as PNG to keep alpha channel
    cv2.imwrite(output_path, rgba)
    print(f"[OK] Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert black background to transparent (batch processing, PNG output)."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing input images"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where processed PNG images will be saved"
    )
    parser.add_argument(
        "--bg_thresh",
        type=int,
        default=5,
        help="RGB threshold for detecting black background. "
             "Pixels with R,G,B < bg_thresh are treated as background. Default: 5"
    )

    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    bg_thresh = args.bg_thresh

    # Supported image extensions
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

    if not os.path.isdir(input_dir):
        print(f"[ERROR] input_dir does not exist or is not a directory: {input_dir}")
        return

    # Iterate through all images in the input directory
    for fname in os.listdir(input_dir):
        if not fname.lower().endswith(exts):
            continue

        in_path = os.path.join(input_dir, fname)

        # Always output PNG (supports transparency)
        base, _ = os.path.splitext(fname)
        out_name = base + ".png"
        out_path = os.path.join(output_dir, out_name)

        process_image(in_path, out_path, bg_thresh=bg_thresh)


if __name__ == "__main__":
    main()
