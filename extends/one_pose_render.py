import numpy as np
from scene.cameras import Camera
from PIL import Image
import torch
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from scene import GaussianModel
from gaussian_renderer import render

# COLMAP行
line = "127 0.22926238101318502 -0.28607131461426677 0.78198511823395012 0.50408455487910486 -0.077584199607372284 -0.96387022733688354 1.4450638294219971 127 train_126.png"
elems = line.strip().split()
img = Image.open("/home/yifan/studium/master_thesis/Anomaly_Detection/SplatPose/MAD-Sim_3dgs/11Pig/train/train_128.png")  # 返回 PIL.Image 对象
img_np = np.array(img)
img_tensor = torch.from_numpy(img_np).float() / 255.0  # 转 float32, 0~1
img_tensor = img_tensor.permute(2,0,1)  # 转成 (C,H,W)
model_dir = "output/pascal_vggt_noBA_11Pig"

parser = ArgumentParser(description="Testing script parameters")
model = ModelParams(parser, sentinel=True)
pipeline = PipelineParams(parser)
parser.add_argument("--iteration", default=-1, type=int)
parser.add_argument("--skip_train", action="store_true")
parser.add_argument("--skip_test", action="store_true")
parser.add_argument("--quiet", action="store_true")
args = get_combined_args(parser)
# print("Rendering " + args.model_path)

dataset = model.extract(args)
pipeline = pipeline.extract(args)
bg_color = [1,1,1]
background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

gaussians = GaussianModel(dataset.sh_degree)
gaussians.load_ply("/home/yifan/studium/master_thesis/gaussian-splatting/output/pascal_vggt_noBA_11Pig/point_cloud/iteration_30000/point_cloud.ply")

IMAGE_ID = int(elems[0])
qw, qx, qy, qz = map(float, elems[1:5])
tx, ty, tz = map(float, elems[5:8])
CAMERA_ID = int(elems[8])
NAME = elems[9]

# 四元数 -> 旋转矩阵
def quat2mat(qw, qx, qy, qz):
    R = np.array([
        [1-2*qy**2-2*qz**2,   2*qx*qy - 2*qz*qw,   2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw,   1-2*qx**2-2*qz**2,   2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw,   2*qy*qz + 2*qx*qw,   1-2*qx**2-2*qy**2]
    ])
    return R

R = quat2mat(qw, qx, qy, qz)
T = np.array([tx, ty, tz])

# 构建 Camera 对象
camera_angle_x = np.deg2rad(60)  # 假设水平 FoV 60度

cur_view = Camera(
    colmap_id=IMAGE_ID,
    R=R,
    T=T,
    FoVx=camera_angle_x,
    FoVy=camera_angle_x,
    image=img_tensor,        # 如果你有对应图像，可填图像数组
    gt_alpha_mask=None,
    image_name=NAME,
    uid=IMAGE_ID
)

rendering = render(cur_view, gaussians, pipeline, background)["render"] # the rendered image from the 3dgs - point cloud from a given view
