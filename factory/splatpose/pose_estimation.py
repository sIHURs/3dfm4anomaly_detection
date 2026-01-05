import os

from gaussian_splatting.utils.loss_utils import l1_loss, ssim
from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.scene.gaussian_model import DiffGaussianModel
from argparse import ArgumentParser
from gaussian_splatting.arguments import ModelParams, PipelineParams
from gaussian_splatting.render import *
from gaussian_splatting.scene.cameras import Camera

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from utils_pose_est import DefectDataset, pose_retrieval_loftr, camera_transf, build_loftr

from torchvision.transforms.functional import to_pil_image

from pathlib import Path
LOFTR_CKPT_PATH = Path(__file__).resolve().parents[1] / "splatpose" /"PAD_utils" / "model" / "indoor_ds_new.ckpt"

classnames = ["01Gorilla", "02Unicorn", "03Mallard", "04Turtle", "05Whale", "06Bird", "07Owl", "08Sabertooth",
              "09Swan", "10Sheep", "11Pig", "12Zalika", "13Pheonix", "14Elephant", "15Parrot", "16Cat", "17Scorpion",
              "18Obesobeso", "19Bear", "20Puppy"]

def main_pose_estimation(cur_class, result_dir, model_dir_location, k=150, verbose=False, data_dir=None, pcd_name="point_cloud.ply", json_name="transforms.json"):
    
    result_dir = result_dir
    output_dir = os.path.join(model_dir_location, "output")
    model_dir = output_dir if os.path.isdir(output_dir) else model_dir_location
    data_dir = "MAD-Sim/" if data_dir is None else data_dir
    trainset = DefectDataset(data_dir, cur_class, "train", True, True, gt_file=json_name)

    # train_imgs = torch.cat([a[0][None,...] for a in trainset], dim=0)
    # train_poses = np.concatenate([np.array(a["transform_matrix"])[None,...] for a in trainset.camera_transforms["frames"]])
    # train_imgs = torch.movedim(torch.nn.functional.interpolate(train_imgs, (400,400)), 1, 3).numpy()
    train_imgs = torch.cat([a[0][None, ...] for a in trainset], dim=0)          # N, C, H, W
    train_imgs = torch.nn.functional.interpolate(train_imgs, (400, 400))        # N,C,400,400
    train_imgs = torch.movedim(train_imgs, 1, 3).contiguous()                   # N,400,400,C

    train_imgs = (train_imgs * 255).to(device="cuda", dtype=torch.float16)
    train_poses = np.stack([np.array(f["transform_matrix"]) for f in trainset.camera_transforms["frames"]], axis=0)

    testset = DefectDataset(data_dir, cur_class, "test", True, True, gt_file=json_name)
    camera_angle_x = trainset.camera_angle

    # Set up command line argument parser
    eval_args = ["-w", "--eval", "-m", model_dir]
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser, my_cmdline=eval_args)


    dataset = model.extract(args)
    pipeline = pipeline.extract(args)
    bg_color = [1,1,1]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    if verbose:
        save_to = os.path.join(result_dir, "3dgs_imgs")
        os.makedirs(save_to, exist_ok=True)

    pose_start = torch.cuda.Event(enable_timing=True)
    pose_end = torch.cuda.Event(enable_timing=True)

    loftr_start = torch.cuda.Event(enable_timing=True)
    loftr_end   = torch.cuda.Event(enable_timing=True)

    normal_images = list()
    reference_images = list()
    all_labels = list()
    gt_masks = list()
    pose_times = list()
    loftr_times = list()
    filenames = list()

    matcher = build_loftr(LOFTR_CKPT_PATH)

    print("LOAD 3DGS MODEL")
    cam_transf = camera_transf().to("cuda")
    
    gaussians = DiffGaussianModel(dataset.sh_degree, torch.eye(4, device="cuda"), cam_transf)
    gaussians.load_ply(os.path.join(model_dir,
                                    "point_cloud",
                                    "iteration_" + str(30000),
                                    pcd_name))
    
    print("STARTING POSE ESTIMATION")
    
    for i in tqdm(range(len(testset))):
        cur_path = testset.images[i].split("/")
        filename = f"{cur_path[-2]}_{cur_path[-1]}.png"
        filenames.append(filename)

        set_entry = testset[i]
        
        all_labels.append(set_entry[1])
        
        gt_masks.append(set_entry[2].cpu().numpy())
        # obs_img = torch.movedim(torch.nn.functional.interpolate(set_entry[0][None,...], (400, 400)).squeeze(), 0, 2)
        obs_img = torch.nn.functional.interpolate(set_entry[0][None, ...], (400,400)).squeeze(0)  # C,H,W
        obs_img = torch.movedim(obs_img, 0, 2).contiguous()                                       # H,W,C
        obs_img = (obs_img * 255).to("cuda", dtype=torch.float16)

        loftr_start.record()

        c2w_init_idx = pose_retrieval_loftr(matcher, train_imgs, obs_img, "cuda")
        c2w_init_np = train_poses[c2w_init_idx]

        pose_start.record()
        c2w_init = torch.from_numpy(c2w_init_np).float().to("cuda")

        c2w_init = c2w_init.clone()
        c2w_init[:3, 1:3] *= -1

        w2c = torch.linalg.inv(c2w_init)  # (4,4), CUDA
        R = w2c[:3, :3].T  # (3,3)
        T = w2c[:3, 3]     # (3,)
        c2w_init[:3, :3] = R
        c2w_init[:3, 3] = T

        # cam_transf = camera_transf().to("cuda")
        gaussians.c2w_init = c2w_init
        cam_transf.reset_()
        cam_transf.train()

        optimizer = torch.optim.Adam(cam_transf.parameters(), lr=0.001, betas=(0.9, 0.999))

        # todo: tmp hard coded
        resolution = (800, 800) 
        # new version requires pil image input
        img = set_entry[0]
        if img.ndim == 3 and img.shape[0] in (1,3,4):  # CHW
            pil_img = to_pil_image(img)
        else:  # HWC
            pil_img = to_pil_image(img.permute(2,0,1))

        # fixed camera pose
        cur_view = Camera(colmap_id=123, 
                            R=c2w_init[:3,:3].cpu().numpy(),
                            T=c2w_init[:3,3].cpu().numpy(),
                            FoVx=camera_angle_x, 
                            FoVy=camera_angle_x,
                            image=pil_img, 
                            image_name="aha", 
                            depth_params=None,
                            invdepthmap=None,
                            resolution=resolution,
                            uid=123)
        
        init_image = None
        
        for iters in range(k):
            optimizer.zero_grad()
            
            gaussians.prepare_forward() 
            rendering = render(cur_view, gaussians, pipeline, background)["render"]

            if init_image is None:
                init_image = torch.clone(rendering).cpu().detach()

            gt_image = set_entry[0].to("cuda")
            loss = 0.8 * l1_loss(rendering, gt_image) + 0.2 * (1 - ssim(rendering, gt_image))
            loss.backward()
          
            optimizer.step()
            
            new_lrate = 0.01 * (0.8 ** ((iters + 1) / 100))
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lrate

            if iters == k - 1:
                if verbose:
                    
                    cur_save = os.path.join(save_to, filename.split(".")[0])
                    os.makedirs(cur_save, exist_ok=True)
                    torchvision.utils.save_image(set_entry[0].cpu().detach(), os.path.join(cur_save, "gt.png"))
                    torchvision.utils.save_image(init_image, os.path.join(cur_save, "first_pose.png"))
                    torchvision.utils.save_image(rendering.cpu().detach(), os.path.join(cur_save, "result.png"))

                    diff_raw = (
                        torch.movedim(torch.abs(rendering - set_entry[0].to("cuda")), 0, 2)
                        .sum(dim=2)
                        .cpu()
                        .detach()
                        .numpy()
                    )

                    plt.imsave(
                        os.path.join(cur_save, "diff.png"),
                        diff_raw,
                        cmap="viridis"
                    )
                                        
                    fig, axs = plt.subplots(2,2, figsize=(10, 6.4))
                    axs[0, 0].set_title("original image"), axs[0, 1].set_title("first pose")
                    axs[1,0].set_title(f"iteration {iters}"), axs[1,1].set_title(f"diff")
                    axs[0,0].imshow(torch.movedim(set_entry[0], 0, 2).cpu().detach())
                    axs[0,1].imshow(torch.movedim(init_image, 0, 2))
                    axs[1,0].imshow(torch.movedim(rendering, 0, 2).cpu().detach())
                    axs[1,1].imshow(torch.movedim(torch.abs(rendering - set_entry[0].to("cuda")), 0, 2).sum(dim=2).cpu().detach())
                    

                    fig.savefig(os.path.join(save_to, filename))
                    plt.close(fig)

                normal_images.append(set_entry[0].cpu().detach())
                reference_images.append(rendering.cpu().detach())

        pose_end.record()
        loftr_end.record()
        torch.cuda.synchronize()
        pose_times.append(pose_start.elapsed_time(pose_end))
        loftr_times.append(loftr_start.elapsed_time(loftr_end))
    
    assert len(normal_images) == len(reference_images) == len(testset), f"Wrongly sized sets!" \
                                                                         f"{len(normal_images)}. {len(reference_images)}. {len(testset)}"
    assert len(normal_images) == len(gt_masks), f"Wrongly sized sets! {len(normal_images)}. {len(gt_masks)}"
    return normal_images, reference_images, all_labels, gt_masks, pose_times, loftr_times, filenames
