#!/usr/bin/env python3
"""
Run BiRefNet on a folder of images, optionally refine foreground colors (FB blur fusion),
save:
  1) masked images with WHITE background
  2) binary masks (optionally hole-filled)

Example:
  python run_birefnet_mask_whitebg_refine.py \
    --input_dir /path/to/images \
    --masked_subdir masked_white \
    --masks_subdir masks \
    --threshold 128 \
    --infer_size 1024 \
    --fill_holes \
    --refine_foreground \
    --refine_r 90 \
    --fp16 \
    --overwrite
"""

import argparse
from pathlib import Path
import os

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageSegmentation


# -----------------------------
# Utils: RGBA -> RGB on white
# -----------------------------
def rgba2rgb(img: Image.Image) -> Image.Image:
    img = img.convert("RGBA")
    bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
    return Image.alpha_composite(bg, img).convert("RGB")


# -----------------------------
# Foreground refinement (same idea as HF Space demo)
# -----------------------------
def mean_blur(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """
    equivalent to cv2.blur for torch tensor
    x: [B, C, H, W]
    """
    if kernel_size % 2 == 0:
        pad_l = kernel_size // 2 - 1
        pad_r = kernel_size // 2
        pad_t = kernel_size // 2 - 1
        pad_b = kernel_size // 2
    else:
        pad_l = pad_r = pad_t = pad_b = kernel_size // 2

    x_padded = torch.nn.functional.pad(x, (pad_l, pad_r, pad_t, pad_b), mode="replicate")
    return torch.nn.functional.avg_pool2d(
        x_padded,
        kernel_size=(kernel_size, kernel_size),
        stride=1,
        count_include_pad=False,
    )


def FB_blur_fusion_foreground_estimator_gpu(
    image: torch.Tensor, FG: torch.Tensor, B: torch.Tensor, alpha: torch.Tensor, r: int = 90
):
    """
    image/FG/B/alpha: [B,C,H,W] in [0,1]
    """
    as_dtype = lambda x, dtype: x.to(dtype) if x.dtype != dtype else x

    input_dtype = image.dtype
    image = as_dtype(image, torch.float32)
    FG = as_dtype(FG, torch.float32)
    B = as_dtype(B, torch.float32)
    alpha = as_dtype(alpha, torch.float32)

    blurred_alpha = mean_blur(alpha, kernel_size=r)

    blurred_FGA = mean_blur(FG * alpha, kernel_size=r)
    blurred_FG = blurred_FGA / (blurred_alpha + 1e-5)

    blurred_B1A = mean_blur(B * (1 - alpha), kernel_size=r)
    blurred_B = blurred_B1A / ((1 - blurred_alpha) + 1e-5)

    FG_output = blurred_FG + alpha * (image - alpha * blurred_FG - (1 - alpha) * blurred_B)
    FG_output = torch.clamp(FG_output, 0, 1)

    return as_dtype(FG_output, input_dtype), as_dtype(blurred_B, input_dtype)


def FB_blur_fusion_foreground_estimator_gpu_2(image: torch.Tensor, alpha: torch.Tensor, r: int = 90) -> torch.Tensor:
    # two-pass trick from BiRefNet community
    FG, blur_B = FB_blur_fusion_foreground_estimator_gpu(image, image, image, alpha, r)
    return FB_blur_fusion_foreground_estimator_gpu(image, FG, blur_B, alpha, r=6)[0]


def FB_blur_fusion_foreground_estimator_cpu(image: np.ndarray, FG: np.ndarray, B: np.ndarray, alpha: np.ndarray, r=90):
    """
    image/FG/B: [H,W,3] float in [0,1]
    alpha: [H,W,1] float in [0,1]
    """
    blurred_alpha = cv2.blur(alpha, (r, r))[:, :, None]

    blurred_FGA = cv2.blur(FG * alpha, (r, r))
    blurred_FG = blurred_FGA / (blurred_alpha + 1e-5)

    blurred_B1A = cv2.blur(B * (1 - alpha), (r, r))
    blurred_B = blurred_B1A / ((1 - blurred_alpha) + 1e-5)

    FG = blurred_FG + alpha * (image - alpha * blurred_FG - (1 - alpha) * blurred_B)
    FG = np.clip(FG, 0, 1)
    return FG, blurred_B


def FB_blur_fusion_foreground_estimator_cpu_2(image: np.ndarray, alpha: np.ndarray, r=90) -> np.ndarray:
    alpha = alpha[:, :, None]
    FG, blur_B = FB_blur_fusion_foreground_estimator_cpu(image, image, image, alpha, r)
    return FB_blur_fusion_foreground_estimator_cpu(image, FG, blur_B, alpha, r=6)[0]


def refine_foreground(image_pil: Image.Image, alpha_pil: Image.Image, r: int, device: torch.device) -> Image.Image:
    """
    image_pil: RGB PIL
    alpha_pil: L/gray PIL in [0..255] (soft alpha)
    Returns refined foreground RGB PIL.
    """
    if alpha_pil.size != image_pil.size:
        alpha_pil = alpha_pil.resize(image_pil.size, resample=Image.BILINEAR)

    if device.type == "cuda":
        img_t = transforms.functional.to_tensor(image_pil).float().to(device)  # [3,H,W] 0..1
        a_t = transforms.functional.to_tensor(alpha_pil).float().to(device)    # [1,H,W] 0..1
        img_t = img_t.unsqueeze(0)
        a_t = a_t.unsqueeze(0)

        fg = FB_blur_fusion_foreground_estimator_gpu_2(img_t, a_t, r=r)  # [1,3,H,W]
        fg = fg.squeeze(0).clamp(0, 1)

        fg_u8 = (fg * 255.0).to(torch.uint8).permute(1, 2, 0).contiguous().cpu().numpy()
        return Image.fromarray(fg_u8)
    else:
        img = np.array(image_pil, dtype=np.float32) / 255.0
        a = np.array(alpha_pil, dtype=np.float32) / 255.0
        fg = FB_blur_fusion_foreground_estimator_cpu_2(img, a, r=r)  # [H,W,3]
        fg_u8 = (fg * 255.0).astype(np.uint8)
        return Image.fromarray(fg_u8)


def composite_on_white(image_rgb: Image.Image, alpha_pil: Image.Image) -> Image.Image:
    """
    Composite RGB foreground over white background with soft alpha.
    """
    if alpha_pil.size != image_rgb.size:
        alpha_pil = alpha_pil.resize(image_rgb.size, resample=Image.BILINEAR)

    fg_rgba = image_rgb.copy().convert("RGBA")
    fg_rgba.putalpha(alpha_pil.convert("L"))

    bg = Image.new("RGBA", image_rgb.size, (255, 255, 255, 255))
    out = Image.alpha_composite(bg, fg_rgba).convert("RGB")
    return out


# -----------------------------
# BiRefNet inference
# -----------------------------
def load_birefnet(device: torch.device, weights: str, fp16: bool):
    model = AutoModelForImageSegmentation.from_pretrained(weights, trust_remote_code=True).to(device)
    model.eval()
    if fp16 and device.type == "cuda":
        model.half()
    else:
        model.float()
    return model


@torch.no_grad()
def birefnet_predict_alpha(
    model,
    image_pil: Image.Image,
    device: torch.device,
    infer_size: int = 1024,
    fp16: bool = False,
) -> Image.Image:
    """
    Returns: alpha PIL (L mode) 0..255, resized back to original image size.
    """
    transform_image = transforms.Compose(
        [
            transforms.Resize((infer_size, infer_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    inp = transform_image(image_pil).unsqueeze(0).to(device)
    if fp16 and device.type == "cuda":
        inp = inp.half()
    else:
        inp = inp.float()

    preds = model(inp)[-1].sigmoid().detach().float().cpu()  # keep alpha in float32 for stability
    pred = preds[0].squeeze()  # [h,w] in [0,1]

    alpha_pil = transforms.ToPILImage()(pred)  # L, 0..255
    alpha_pil = alpha_pil.resize(image_pil.size, resample=Image.BILINEAR)
    return alpha_pil


def alpha_to_binary_mask(alpha_pil: Image.Image, threshold: int) -> np.ndarray:
    """
    alpha_pil: 0..255
    returns uint8 mask01 in {0,1}, shape (H,W)
    """
    a = np.array(alpha_pil.convert("L"), dtype=np.uint8)
    return (a > threshold).astype(np.uint8)


def fill_holes(mask01: np.ndarray) -> np.ndarray:
    """
    Fill holes inside foreground (mask==1).
    mask01: uint8 {0,1} (H,W)
    """
    m = (mask01.astype(np.uint8) * 255)
    h, w = m.shape[:2]
    flood = m.copy()
    ff_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, ff_mask, seedPoint=(0, 0), newVal=255)
    flood_inv = cv2.bitwise_not(flood)
    filled = cv2.bitwise_or(m, flood_inv)
    return (filled > 0).astype(np.uint8)


def save_mask(mask01: np.ndarray, path: Path):
    Image.fromarray((mask01 * 255).astype(np.uint8)).save(path)


# -----------------------------
# CLI
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="BiRefNet masking for a folder (white background output + binary masks).")
    p.add_argument("--input_dir", type=str, required=True, help="Folder containing images (e.g. .png)")
    p.add_argument("--masked_subdir", type=str, default="masked_white", help="Subfolder to save masked images")
    p.add_argument("--masks_subdir", type=str, default="masks", help="Subfolder to save binary masks")
    p.add_argument("--threshold", type=int, default=128, help="Binarization threshold in [0..255] (128 ~= 0.5)")
    p.add_argument("--infer_size", type=int, default=1024, help="Inference resize (square), default 1024")
    p.add_argument("--pattern", type=str, default="*.png", help="Glob pattern, default '*.png'")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Compute device")

    # new options
    p.add_argument("--weights", type=str, default="ZhengPeng7/BiRefNet", help="HF weights repo id")
    p.add_argument("--fp16", action="store_true", help="Use fp16 for model/inference on CUDA")
    p.add_argument("--fill_holes", action="store_true", help="Fill holes in the binary mask before saving")
    p.add_argument("--refine_foreground", action="store_true", help="Use FB blur fusion refinement for foreground colors")
    p.add_argument("--refine_r", type=int, default=90, help="Refinement blur radius r (default 90)")

    p.add_argument("--soft_subdir", type=str, default="masks_soft", help="Subfolder to save soft alpha masks (0..255).")
    p.add_argument("--save_soft", action="store_true", help="Save soft alpha masks (0..255).")

    return p.parse_args()


def main():
    args = parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Not a directory: {input_dir}")

    out_masked = input_dir / args.masked_subdir
    out_masks = input_dir / args.masks_subdir
    out_masked.mkdir(parents=True, exist_ok=True)
    out_masks.mkdir(parents=True, exist_ok=True)

    if args.device == "cuda":
        device = torch.device("cuda")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[info] device: {device}")
    print(f"[info] input_dir: {input_dir}")
    print(f"[info] save masked -> {out_masked}")
    print(f"[info] save masks  -> {out_masks}")
    print(f"[info] weights: {args.weights}")
    print(f"[info] fp16: {bool(args.fp16)}")
    print(f"[info] threshold: {args.threshold} (threshold/255={args.threshold/255:.3f})")
    print(f"[info] infer_size: {args.infer_size}")
    print(f"[info] fill_holes: {bool(args.fill_holes)}")
    print(f"[info] refine_foreground: {bool(args.refine_foreground)} (r={args.refine_r})")

    model = load_birefnet(device=device, weights=args.weights, fp16=args.fp16)

    imgs = sorted(input_dir.glob(args.pattern))
    if not imgs:
        print(f"[warn] no files match: {args.pattern} in {input_dir}")
        return

    for img_path in imgs:
        out_img_path = out_masked / img_path.name
        out_mask_path = out_masks / img_path.name

        if not args.overwrite and (out_img_path.exists() or out_mask_path.exists()):
            print(f"[skip] {img_path.name} (exists)")
            continue

        try:
            image_raw = Image.open(img_path)
            if image_raw.mode == "RGBA":
                image = rgba2rgb(image_raw)
            else:
                image = image_raw.convert("RGB")

            # 1) predict soft alpha (0..255)
            alpha_pil = birefnet_predict_alpha(
                model=model,
                image_pil=image,
                device=device,
                infer_size=args.infer_size,
                fp16=args.fp16,
            )

            if args.save_soft:
                out_soft = input_dir / args.soft_subdir
                out_soft.mkdir(parents=True, exist_ok=True)
                (out_soft / img_path.name).parent.mkdir(parents=True, exist_ok=True)
                alpha_pil.convert("L").save(out_soft / img_path.name)

            # 2) binary mask (optionally hole-filled) -> save
            mask01 = alpha_to_binary_mask(alpha_pil, threshold=args.threshold)
            if args.fill_holes:
                mask01 = fill_holes(mask01)
            save_mask(mask01, out_mask_path)

            # 3) masked image with white background
            #    - if refine_foreground: refine RGB using SOFT alpha, then composite on white with soft alpha
            #    - else: just composite original image on white with soft alpha
            if args.refine_foreground:
                refined_fg = refine_foreground(image, alpha_pil, r=args.refine_r, device=device)
                masked_white = composite_on_white(refined_fg, alpha_pil)
            else:
                masked_white = composite_on_white(image, alpha_pil)

            masked_white.save(out_img_path)

            if device.type == "cuda":
                torch.cuda.empty_cache()

            print(f"[ok] {img_path.name}")
        except Exception as e:
            print(f"[err] {img_path.name}: {e}")

    print("[done]")


if __name__ == "__main__":
    main()