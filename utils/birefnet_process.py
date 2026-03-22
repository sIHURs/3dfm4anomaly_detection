#!/usr/bin/env python3
"""
Run BiRefNet on a folder of PNG images, generate binary masks, and save
masked images with WHITE background + masks into subfolders.

Example:
  python run_birefnet_mask_whitebg.py \
    --input_dir /path/to/images \
    --masked_subdir masked_white \
    --masks_subdir masks \
    --threshold 128 \
    --overwrite
"""

import argparse
from pathlib import Path
import cv2

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageSegmentation


def load_birefnet(device: torch.device):
    model = AutoModelForImageSegmentation.from_pretrained(
        "ZhengPeng7/BiRefNet",
        trust_remote_code=True,
    ).to(device)
    model.eval()
    return model


@torch.no_grad()
def birefnet_mask(
    model,
    image_pil: Image.Image,
    device: torch.device,
    threshold: int,
    infer_size: int = 1024,
) -> np.ndarray:
    """
    Returns: mask uint8 in {0,1}, shape (H, W) matching original image size.
    """
    transform_image = transforms.Compose(
        [
            transforms.Resize((infer_size, infer_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    inp = transform_image(image_pil).unsqueeze(0).to(device)

    # BiRefNet (HF remote code) typically returns multi-scale outputs; take the last one.
    preds = model(inp)[-1].sigmoid().cpu()
    pred = preds[0].squeeze()  # [h,w]

    # Convert prob map [0,1] -> PIL (uint8 0..255), then resize back to original size
    pred_pil = transforms.ToPILImage()(pred)
    pred_pil = pred_pil.resize(image_pil.size, resample=Image.BILINEAR)

    mask_np = np.array(pred_pil)  # uint8 0..255
    mask01 = (mask_np > threshold).astype(np.uint8)
    return mask01


def apply_mask_white_bg(image_pil: Image.Image, mask01: np.ndarray) -> Image.Image:
    """
    image_pil -> RGB output with white background (255,255,255).
    mask01: (H,W) in {0,1} where 1=foreground.
    """
    img = np.array(image_pil.convert("RGB"))  # (H,W,3)
    if mask01.shape != img.shape[:2]:
        raise ValueError(f"Mask shape {mask01.shape} != image shape {img.shape[:2]}")

    out = img.copy()
    out[mask01 == 0] = 255  # white background
    return Image.fromarray(out.astype(np.uint8))


def save_mask(mask01: np.ndarray, path: Path):
    Image.fromarray((mask01 * 255).astype(np.uint8)).save(path)


def fill_holes(mask01: np.ndarray) -> np.ndarray:
    """
    Fill holes inside foreground (mask==1).
    mask01: uint8 {0,1} (H,W)
    return: uint8 {0,1} (H,W)
    """
    m = (mask01.astype(np.uint8) * 255)

    h, w = m.shape[:2]
    flood = m.copy()

    # floodFill 需要 (h+2, w+2) 的 mask
    ff_mask = np.zeros((h + 2, w + 2), np.uint8)

    # 从(0,0)开始灌背景（假设边界背景是0）
    cv2.floodFill(flood, ff_mask, seedPoint=(0, 0), newVal=255)

    # flood 里被灌成255的是“外部背景”，取反得到“洞+前景”
    flood_inv = cv2.bitwise_not(flood)

    # 原前景 OR 洞 => 填洞后的前景
    filled = cv2.bitwise_or(m, flood_inv)

    return (filled > 0).astype(np.uint8)


def parse_args():
    p = argparse.ArgumentParser(description="BiRefNet masking for a folder of PNGs (white background output).")
    p.add_argument("--input_dir", type=str, required=True, help="Folder containing .png images")
    p.add_argument("--masked_subdir", type=str, default="masked_white", help="Subfolder to save masked images")
    p.add_argument("--masks_subdir", type=str, default="masks", help="Subfolder to save binary masks")
    p.add_argument("--threshold", type=int, default=128, help="Binarization threshold in [0..255] (128 ~= 0.5)")
    p.add_argument("--infer_size", type=int, default=1024, help="Inference resize (square), default 1024")
    p.add_argument("--pattern", type=str, default="*.png", help="Glob pattern, default '*.png'")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Compute device")
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
    print(f"[info] threshold: {args.threshold} (threshold/255={args.threshold/255:.3f})")
    print(f"[info] infer_size: {args.infer_size}")

    model = load_birefnet(device)
    model = model.float()

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
            image = Image.open(img_path).convert("RGB")
            mask01 = birefnet_mask(
                model=model,
                image_pil=image,
                device=device,
                threshold=args.threshold,
                infer_size=args.infer_size,
            )
            mask01 = fill_holes(mask01)
            masked = apply_mask_white_bg(image, mask01)
            masked.save(out_img_path)
            save_mask(mask01, out_mask_path)

            print(f"[ok] {img_path.name}")
        except Exception as e:
            print(f"[err] {img_path.name}: {e}")
            

    print("[done]")


if __name__ == "__main__":
    main()