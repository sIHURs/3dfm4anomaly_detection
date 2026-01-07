#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
from argparse import ArgumentParser

import torch
import torchvision
from tqdm import tqdm
from torchvision.utils import save_image

from factory.gaussian_splatting.utils.loss_utils import l1_loss, ssim
from factory.gaussian_splatting.gaussian_renderer import render
from factory.gaussian_splatting.scene import Scene, GaussianModel
from factory.gaussian_splatting.utils.general_utils import safe_state
from factory.gaussian_splatting.utils.image_utils import psnr
from factory.gaussian_splatting.arguments import ModelParams, PipelineParams

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except Exception:
    FUSED_SSIM_AVAILABLE = False


def _list_images_sorted(img_dir: str):
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
    files = [f for f in os.listdir(img_dir) if f.lower().endswith(exts)]
    files.sort()
    return files


def _load_image_chw_float01(path: str, device: str = "cuda"):
    """
    Load image as float tensor in [0,1], shape (C,H,W), on device.
    Uses torchvision.io.read_image (returns uint8 CHW).
    """
    x = torchvision.io.read_image(path)  # uint8, CHW
    x = x.float() / 255.0
    if x.ndim == 2:
        x = x.unsqueeze(0)
    # if RGBA, drop alpha for metrics (optional)
    if x.shape[0] == 4:
        x = x[:3]
    return x.to(device)


@torch.no_grad()
def evaluate_ply(
    dataset,
    pipe,
    model_iteration: int,
    save_renders: bool = False,
    max_views: int | None = None,
    external_image_dir: str | None = None,
    external_match: str = "sorted",
):
    """
    Evaluate a trained 3DGS Gaussian model (point_cloud.ply) on all train cameras.

    Args:
        dataset: ModelParams.extract(args)
        pipe: PipelineParams.extract(args)
        model_iteration: e.g. 30000
        save_renders: whether to save rendered images
        max_views: if not None, only evaluate on first N train views
        external_image_dir: if provided, load GT images from this directory instead of dataset images
        external_match:
            - "sorted": match GT images by sorted order (idx)
            - "by_image_name": match by viewpoint.image_name (basename), with common extension fallbacks
    """
    # --- set up Gaussians & Scene (cameras) ---
    gaussians = GaussianModel(dataset.sh_degree, "adam")  # optimizer type irrelevant for eval
    scene = Scene(dataset, gaussians)

    # --- load trained PLY ---
    ply_path = os.path.join(
        scene.model_path,
        "point_cloud",
        f"iteration_{model_iteration}",
        "point_cloud_clean_t0.100_gaussiansOpt.ply",
    )
    if not os.path.isfile(ply_path):
        raise FileNotFoundError(f"PLY not found: {ply_path}")

    print(f"[EVAL] Loading Gaussians from: {ply_path}")
    gaussians.load_ply(ply_path)

    # --- background color ---
    bg_color = [1, 1, 1]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    train_cams = scene.getTrainCameras()
    if max_views is not None:
        train_cams = train_cams[:max_views]
    print(f"[EVAL] Number of training views: {len(train_cams)}")

    # --- external GT prep (optional) ---
    ext_files = None
    if external_image_dir is not None:
        external_image_dir = os.path.abspath(external_image_dir)
        if not os.path.isdir(external_image_dir):
            raise FileNotFoundError(f"external_image_dir not found: {external_image_dir}")

        if external_match == "sorted":
            ext_files = _list_images_sorted(external_image_dir)
            if max_views is not None:
                ext_files = ext_files[:max_views]
            if len(ext_files) < len(train_cams):
                raise RuntimeError(
                    f"Not enough images in external_image_dir. "
                    f"Need >= {len(train_cams)}, got {len(ext_files)}"
                )
            print(f"[EVAL] Using external GT images (sorted) from: {external_image_dir}")

        elif external_match == "by_image_name":
            # build a map from basename (without ext) to full filename
            all_files = _list_images_sorted(external_image_dir)
            stem2file = {}
            for f in all_files:
                stem = os.path.splitext(f)[0]
                stem2file[stem] = f
            ext_files = stem2file
            print(f"[EVAL] Using external GT images (by_image_name) from: {external_image_dir}")

        else:
            raise ValueError(f"Unknown external_match: {external_match}")

    # directory to save renders (optional)
    if save_renders:
        out_dir = os.path.join(scene.model_path, f"eval_renders_train_iter_{model_iteration}")
        os.makedirs(out_dir, exist_ok=True)
        print(f"[EVAL] Saving renders to: {out_dir}")

    # --- metric accumulators ---
    total_l1 = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    num_views = 0

    for idx, viewpoint in enumerate(tqdm(train_cams, desc="Evaluating train views")):
        # render current view
        render_pkg = render(
            viewpoint,
            gaussians,
            pipe,
            background,
            use_trained_exp=dataset.train_test_exp,
            separate_sh=False,
        )
        image = torch.clamp(render_pkg["render"], 0.0, 1.0)

        # choose GT image
        if external_image_dir is None:
            gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
        else:
            if external_match == "sorted":
                img_path = os.path.join(external_image_dir, ext_files[idx])
            else:  # by_image_name
                # viewpoint.image_name is usually a filename-like string; try to match its stem
                stem = os.path.splitext(os.path.basename(viewpoint.image_name))[0]
                if stem not in ext_files:
                    # try some common alternate stems (sometimes image_name contains folders)
                    raise FileNotFoundError(
                        f"Could not match viewpoint.image_name='{viewpoint.image_name}' "
                        f"(stem='{stem}') in external_image_dir='{external_image_dir}'."
                    )
                img_path = os.path.join(external_image_dir, ext_files[stem])

            gt_image = _load_image_chw_float01(img_path, device="cuda")
            gt_image = torch.clamp(gt_image, 0.0, 1.0)

        # exp case: only use half (like in training_report)
        if dataset.train_test_exp:
            image = image[..., image.shape[-1] // 2 :]
            gt_image = gt_image[..., gt_image.shape[-1] // 2 :]

        # metrics
        l1_val = l1_loss(image, gt_image).mean().double()
        psnr_val = psnr(image, gt_image).mean().double()
        if FUSED_SSIM_AVAILABLE:
            ssim_val = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0)).double()
        else:
            ssim_val = ssim(image, gt_image).double()

        total_l1 += l1_val.item()
        total_psnr += psnr_val.item()
        total_ssim += ssim_val.item()
        num_views += 1

        # optionally save renders
        if save_renders:
            # keep name stable
            save_name = f"train_{idx:04d}_{os.path.basename(viewpoint.image_name)}"
            if not save_name.lower().endswith(".png"):
                save_name += ".png"
            save_path = os.path.join(out_dir, save_name)
            save_image(image, save_path)

    # --- final averages ---
    avg_l1 = total_l1 / max(1, num_views)
    avg_psnr = total_psnr / max(1, num_views)
    avg_ssim = total_ssim / max(1, num_views)

    print("\n========== 3DGS Train Set Evaluation ==========")
    print(f"Iteration:       {model_iteration}")
    print(f"#Train views:    {num_views}")
    if external_image_dir is None:
        print("GT source:       dataset (viewpoint.original_image)")
    else:
        print(f"GT source:       external_image_dir={external_image_dir} (match={external_match})")
    print(f"Average L1:      {avg_l1:.6f}")
    print(f"Average PSNR:    {avg_psnr:.3f} dB")
    print(f"Average SSIM:    {avg_ssim:.4f}")
    print("===============================================")

    return {
        "avg_l1": avg_l1,
        "avg_psnr": avg_psnr,
        "avg_ssim": avg_ssim,
        "num_views": num_views,
    }


if __name__ == "__main__":
    parser = ArgumentParser(description="Evaluate a trained 3DGS model on train views (optionally with external GT images)")
    lp = ModelParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument(
        "--model_iteration",
        type=int,
        default=30_000,
        help="Which iteration of point_cloud.ply to load (e.g. 30000)",
    )
    parser.add_argument(
        "--max_views",
        type=int,
        default=None,
        help="If set, only evaluate on first N train views",
    )
    parser.add_argument(
        "--save_renders",
        action="store_true",
        help="If set, save rendered train views to disk",
    )
    parser.add_argument(
        "--external_image_dir",
        type=str,
        default=None,
        help="Optional directory to load GT images from instead of dataset images",
    )
    parser.add_argument(
        "--external_match",
        type=str,
        default="sorted",
        choices=["sorted", "by_image_name"],
        help="How to match external GT images to cameras",
    )

    args = parser.parse_args(sys.argv[1:])

    # keep the same default model_path logic as training scripts often do
    # (if your ModelParams already handles model_path, this line is harmless)
    if getattr(args, "model_path", "") == "":
        args.model_path = os.path.join(args.source_path, "output")

    print("Evaluating model in:", args.model_path)

    safe_state(False)

    dataset = lp.extract(args)
    pipe = pp.extract(args)

    evaluate_ply(
        dataset=dataset,
        pipe=pipe,
        model_iteration=args.model_iteration,
        save_renders=args.save_renders,
        max_views=args.max_views,
        external_image_dir=args.external_image_dir,
        external_match=args.external_match,
    )

    print("\nEvaluation complete.")
