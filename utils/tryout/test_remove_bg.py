import os
import cv2
import argparse
import numpy as np


def remove_white_bg_smooth(
    input_dir: str,
    output_dir: str,
    white_thresh: int = 245,
    dist_thresh: float = 18.0,
    open_ksize: int = 3,
    close_ksize: int = 5,
    feather: int = 2,
    dematte: bool = True,
):
    """
    Remove white/light background and export transparent PNGs with smoother edges.

    Strategy:
      1) Build background mask using distance-to-white and a brightness constraint.
      2) Morphological open/close to remove speckles and fill tiny holes.
      3) Feather alpha (Gaussian blur) to smooth jagged edges.
      4) Optional dematte to reduce white halo on semi-transparent edges.

    Args:
        input_dir:  Directory containing input images (.jpg/.jpeg/.png).
        output_dir: Directory to save output PNGs with alpha.
        white_thresh: Pixel is "white candidate" if all RGB >= this value.
        dist_thresh: Distance-to-white threshold (smaller = stricter, larger = more aggressive).
        open_ksize: Kernel size for MORPH_OPEN (remove small noise). Use 0/1 to disable.
        close_ksize: Kernel size for MORPH_CLOSE (fill small holes). Use 0/1 to disable.
        feather: Gaussian blur radius for alpha smoothing (0 disables).
        dematte: If True, remove white fringe by un-matting against white background.
    """
    os.makedirs(output_dir, exist_ok=True)
    valid_ext = (".png", ".jpg", ".jpeg")

    def make_kernel(k: int):
        k = int(k)
        if k <= 1:
            return None
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    k_open = make_kernel(open_ksize)
    k_close = make_kernel(close_ksize)

    files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(valid_ext)])
    if not files:
        print(f"No images found in: {input_dir}")
        return

    for fname in files:
        in_path = os.path.join(input_dir, fname)
        bgr = cv2.imread(in_path, cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"Failed to read, skipped: {in_path}")
            continue

        # --- 1) Background mask: distance to white in RGB ---
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        white = np.array([255.0, 255.0, 255.0], dtype=np.float32)

        dist = np.linalg.norm(rgb - white, axis=2)  # [0..~441]
        mask_by_dist = dist <= float(dist_thresh)

        # Extra constraint: very bright pixels are likely background
        mask_by_brightness = np.all(rgb >= float(white_thresh), axis=2)

        # Background mask: 255 = background, 0 = foreground
        bg_mask = (mask_by_dist | mask_by_brightness).astype(np.uint8) * 255

        # --- 2) Morphology cleanup ---
        if k_open is not None:
            bg_mask = cv2.morphologyEx(bg_mask, cv2.MORPH_OPEN, k_open)
        if k_close is not None:
            bg_mask = cv2.morphologyEx(bg_mask, cv2.MORPH_CLOSE, k_close)

        # --- 3) Alpha channel (foreground opaque, background transparent) ---
        alpha = 255 - bg_mask

        # Feather edges by blurring alpha
        if feather and feather > 0:
            k = int(2 * feather + 1)  # odd kernel size
            alpha = cv2.GaussianBlur(alpha, (k, k), 0)

        # --- 4) Compose BGRA ---
        bgra = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
        bgra[:, :, 3] = alpha

        # --- 5) Dematte to reduce white fringe ---
        if dematte:
            a = (alpha.astype(np.float32) / 255.0)[..., None]  # HxWx1
            eps = 1e-6

            bgr_f = bgr.astype(np.float32)
            white_bgr = np.array([255.0, 255.0, 255.0], dtype=np.float32)[None, None, :]

            # observed = true*a + white*(1-a)  => true = (observed - white*(1-a)) / max(a, eps)
            true = (bgr_f - white_bgr * (1.0 - a)) / np.maximum(a, eps)
            true = np.clip(true, 0, 255).astype(np.uint8)

            bgra[:, :, 0:3] = true

        out_name = os.path.splitext(fname)[0] + ".png"
        out_path = os.path.join(output_dir, out_name)
        cv2.imwrite(out_path, bgra)

        print(f"Processed: {fname} -> {out_name}")

    print("All images have been converted.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Remove white/light background and export transparent PNGs (smooth edges + dematte)."
    )
    parser.add_argument("--input_dir", type=str, required=True, help="Input image directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for PNG with alpha")

    parser.add_argument("--white_thresh", type=int, default=245,
                        help="RGB brightness threshold (all channels >= this => background candidate)")
    parser.add_argument("--dist_thresh", type=float, default=18.0,
                        help="Distance-to-white threshold (smaller=stricter, larger=more aggressive)")

    parser.add_argument("--open_ksize", type=int, default=3,
                        help="Kernel size for MORPH_OPEN (<=1 disables)")
    parser.add_argument("--close_ksize", type=int, default=5,
                        help="Kernel size for MORPH_CLOSE (<=1 disables)")

    parser.add_argument("--feather", type=int, default=2,
                        help="Alpha feather radius (0 disables)")
    parser.add_argument("--dematte", action="store_true",
                        help="Enable dematte (reduce white halo on edges)")
    parser.add_argument("--no_dematting", action="store_true",
                        help="Disable dematte (overrides --dematte)")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    dematte_flag = True if args.dematte else False
    if args.no_dematting:
        dematte_flag = False

    remove_white_bg_smooth(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        white_thresh=args.white_thresh,
        dist_thresh=args.dist_thresh,
        open_ksize=args.open_ksize,
        close_ksize=args.close_ksize,
        feather=args.feather,
        dematte=dematte_flag,
    )

'''
python remove_white_bg.py \
  --input_dir scripts/demo_PIAD_Sim_vggt_3dgs/motor/images_bg \
  --output_dir scripts/demo_PIAD_Sim_vggt_3dgs/motor/images \
  --dist_thresh 22 \
  --white_thresh 245 \
  --open_ksize 3 \
  --close_ksize 7 \
  --feather 2 \
  --dematte
  '''
