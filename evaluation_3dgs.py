import os
import sys
from argparse import ArgumentParser

import torch
from tqdm import tqdm
from torchvision.utils import save_image

from gaussian_splatting.utils.loss_utils import l1_loss, ssim
from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.scene import Scene, GaussianModel
from gaussian_splatting.utils.general_utils import safe_state
from gaussian_splatting.utils.image_utils import psnr
from gaussian_splatting.arguments import ModelParams, PipelineParams

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except Exception:
    FUSED_SSIM_AVAILABLE = False


@torch.no_grad()
def evaluate_ply(dataset, pipe, model_iteration, save_renders=False, max_views=None):
    """
    Evaluate a trained 3DGS Gaussian model (point_cloud.ply) on all train cameras.

    Args:
        dataset:      ModelParams.extract(args)
        pipe:         PipelineParams.extract(args)
        model_iteration: int, e.g. 30000 → will load
                         <model_path>/point_cloud/iteration_30000/point_cloud.ply
        save_renders: whether to save rendered images
        max_views:    if not None, only evaluate on first N train views
    """

    # --- set up Gaussians & Scene (cameras) ---
    # optimizer_type is irrelevant for evaluation; we just give a dummy string
    gaussians = GaussianModel(dataset.sh_degree, "adam")
    scene = Scene(dataset, gaussians)

    # --- load trained PLY ---
    ply_path = os.path.join(
        scene.model_path,
        "point_cloud",
        f"iteration_{model_iteration}",
        "point_cloud.ply",
    )
    if not os.path.isfile(ply_path):
        raise FileNotFoundError(f"PLY not found: {ply_path}")

    print(f"[EVAL] Loading Gaussians from: {ply_path}")
    gaussians.load_ply(ply_path)

    # --- background color ---
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    train_cams = scene.getTrainCameras()
    if max_views is not None:
        train_cams = train_cams[:max_views]

    print(f"[EVAL] Number of training views: {len(train_cams)}")

    # --- metric accumulators ---
    total_l1 = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    num_views = 0

    # directory to save renders (optional)
    if save_renders:
        out_dir = os.path.join(scene.model_path, f"eval_renders_train_iter_{model_iteration}")
        os.makedirs(out_dir, exist_ok=True)
        print(f"[EVAL] Saving renders to: {out_dir}")

    for idx, viewpoint in enumerate(tqdm(train_cams, desc="Evaluating train views")):
        # render current view
        render_pkg = render(
            viewpoint,
            gaussians,
            pipe,
            background,
            use_trained_exp=dataset.train_test_exp,
            separate_sh=False,  # you can set True if using sparse_adam + separate SH
        )
        image = torch.clamp(render_pkg["render"], 0.0, 1.0)
        gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)

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
            save_path = os.path.join(out_dir, f"train_{idx:04d}_{viewpoint.image_name}.png")
            save_image(image, save_path)

    # --- final averages ---
    avg_l1 = total_l1 / max(1, num_views)
    avg_psnr = total_psnr / max(1, num_views)
    avg_ssim = total_ssim / max(1, num_views)

    print("\n========== 3DGS Train Set Evaluation ==========")
    print(f"Iteration:       {model_iteration}")
    print(f"#Train views:    {num_views}")
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
    # Set up parser
    parser = ArgumentParser(description="Evaluation of a trained 3DGS model on train views")
    lp = ModelParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument("--model_iteration", type=int, default=30_000,
                        help="Which iteration of point_cloud.ply to load")
    parser.add_argument("--max_views", type=int, default=None,
                        help="If set, only evaluate on first N train views")
    parser.add_argument("--save_renders", action="store_true",
                        help="If set, save rendered train views to disk")
    args = parser.parse_args(sys.argv[1:])

    # reuse same default model_path logic as your training script
    args.model_path = os.path.join(args.source_path, "output") if args.model_path == "" else args.model_path
    print("Evaluating model in:", args.model_path)

    safe_state(False)
    # torch.autograd.set_detect_anomaly(args.detect_anomaly)

    dataset = lp.extract(args)
    pipe = pp.extract(args)

    evaluate_ply(
        dataset=dataset,
        pipe=pipe,
        model_iteration=args.model_iteration,
        save_renders=args.save_renders,
        max_views=args.max_views,
    )

    print("\nEvaluation complete.")

    