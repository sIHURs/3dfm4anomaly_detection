
from argparse import ArgumentParser
import os
import wandb

import torch
import numpy as np
import random
from torchvision import transforms
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve
from scipy.ndimage import gaussian_filter

# needed for PAD code
from easydict import EasyDict
import yaml
from PIL import Image

from factory.splatpose.pose_estimation import main_pose_estimation
from factory.splatpose.utils_pose_est import ModelHelper, update_config
from factory.splatpose.aupro import calculate_au_pro_au_roc

# solve path problem
from pathlib import Path
PAD_CONFIG_PATH = Path(__file__).resolve().parents[1] / "3dfm4anomaly_detection" / "factory" / "splatpose" /"PAD_utils" / "config_effnet.yaml"


MAD_classnames = ["01Gorilla", "02Unicorn", "03Mallard", "04Turtle", "05Whale", "06Bird", "07Owl", "08Sabertooth",
              "09Swan", "10Sheep", "11Pig", "12Zalika", "13Pheonix", "14Elephant", "15Parrot", "16Cat", "17Scorpion",
              "18Obesobeso", "19Bear", "20Puppy"]

PIAD_CL_classnames = ["Axletree", "Box", "Can", "Chain", "Gear", "Keyring", "Motor", "Parts", "Picker", "Section", "Shaft",
                      "Spray_can", "Spring", "Sprockets"]

pre_parser = ArgumentParser(description="Parameters of the LEGO training run")
pre_parser.add_argument("-k", metavar="K", type=int, help="number of pose estimation steps", default=175)
pre_parser.add_argument("-c", "--classname", metavar="c", type=str, help="current class to run experiments on",
                        default="01Gorilla")
pre_parser.add_argument("-w", "--use_wandb", type=int, help="the wandb to use", default=0)
pre_parser.add_argument("-p", "--prefix", metavar="pf", type=str, help="prefix for the wandb run name", default="to_delete")
pre_parser.add_argument("--seed", type=int, help="seed for random behavior", default=0)
pre_parser.add_argument("--loftr_batch", type=int, help="batch size for loftr pose retrieval", default=32)
pre_parser.add_argument("--loftr_resolution", type=tuple, help="images resolution for loftr pose retrieval", default=(128,128))
pre_parser.add_argument("--gauss_iters", type=int, help="number of training iterations for 3DGS", default=30000)
pre_parser.add_argument("--wandb", type=int, help="whether we track with wandb", default=1)
# pre_parser.add_argument("--train", type=int, help="whether we train or look for a saved model", default=1)               
pre_parser.add_argument("-v", "--verbose", type=int, help="verbosity", default=0)                        
pre_parser.add_argument("--data_path", type=str, help="path pointing towards the usable data set", default="MAD-Sim_3dgs/")                        
pre_parser.add_argument("--result", type=str, help="path of output result", default="ad_result")
pre_parser.add_argument("--model_path_splatpose", type=str, help="path of 3dgs output model", default="output")
pre_parser.add_argument("--pcd_name", type=str, help="name of the processed 3dgs poind cloud", default="point_cloud.ply")
pre_parser.add_argument("--json_name", type=str, help="name of the camera pose json file", default="transforms.json")
pre_parser.add_argument("--retrieval_model", type=str, help="model for init c2w", default="loftr")

args = pre_parser.parse_args()

if args.use_wandb:
    wandb.init(
        project="splatpose-anomaly",
        config=vars(args),
    )
else:
    wandb.init(mode="disabled")


def seed_everything(seed: int, deterministic: bool = True):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

seed_everything(args.seed, deterministic=True)

result_dir = os.path.join(args.result, f"results_{args.prefix}_{args.seed}", args.classname)
model_dir = os.path.join(args.model_path_splatpose, args.classname)
data_dir = args.data_path

test_images, reference_images, all_labels, gt_masks, times, total_times, filenames = main_pose_estimation(cur_class=args.classname,
                                                                                    result_dir=result_dir,
                                                                                    model_dir_location=model_dir,
                                                                                    k=args.k, 
                                                                                    verbose=args.verbose,
                                                                                    data_dir=data_dir,
                                                                                    pcd_name=args.pcd_name,
                                                                                    json_name=args.json_name,
                                                                                    loftr_batch=args.loftr_batch,
                                                                                    loftr_resolution=args.loftr_resolution,
                                                                                    retrieval=args.retrieval_model)

# todo: some thing wrong with wandb output
if args.use_wandb:
    pose_time = [[i, float(times[i])] for i in range(len(times))]
    pose_table = wandb.Table(
        data=pose_time,
        columns=["index", "value_ms"]
    )

    pose_plot = wandb.plot.line(
        pose_table,
        x="index",
        y="value_ms",
        title="Pose time per image (ms)"
    )

    wandb.log({
        "timing/pose_time_table": pose_table,
        "timing/pose_time_plot": pose_plot,
    })

if args.use_wandb:
    total_time = [[i, float(total_times[i])] for i in range(len(total_times))]
    total_table = wandb.Table(
        data=total_time,
        columns=["index", "value_ms"]
    )

    total_plot = wandb.plot.line(
        total_table,
        x="index",
        y="value_ms",
        title="Total time per image (ms)"
    )

    wandb.log({
        "timing/total_time_table": total_table,
        "timing/total_time_plot": total_plot,
    })

with open(PAD_CONFIG_PATH) as f:
    mad_config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))
mad_config = update_config(mad_config)
model = ModelHelper(mad_config.net)
model.eval()
model.cuda()


# evaluation Code taken from PAD/MAD data set paper at https://github.com/EricLee0224/PAD
criterion = torch.nn.MSELoss(reduction='none')
tf_img = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

tf_mask = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(224, interpolation=transforms.InterpolationMode.NEAREST),
    ])

test_imgs = list()
score_map_list=list()
scores=list()
pred_list=list()
recon_imgs=list()
with torch.no_grad():
    # todo: use batch, not just single images at once
    for i in range(len(test_images)):
        ref=tf_img(reference_images[i]).unsqueeze(0).cuda()
        rgb=tf_img(test_images[i]).unsqueeze(0).cuda()
        fileId = filenames[i]
        # todo: torch.cat([ref, rgb], dim=0) then send into model, inference only once
        ref_feature=model(ref)
        rgb_feature=model(rgb)
        score = criterion(ref, rgb).sum(1, keepdim=True)
        for i in range(len(ref_feature)):
            
            s_act = ref_feature[i]
            mse_loss = criterion(s_act, rgb_feature[i]).sum(1, keepdim=True)
            score += torch.nn.functional.interpolate(mse_loss, size=224, mode='bilinear', align_corners=False)

        score = score.squeeze(1).cpu().numpy()
        # todo: do gaussian_filter on gpu? - kornia?
        for i in range(score.shape[0]):
            score[i] = gaussian_filter(score[i], sigma=4)

        if args.verbose:
            save_dir = os.path.join(result_dir, "3dgs_imgs", fileId.split(".")[0])
            os.makedirs(save_dir, exist_ok=True)

            vis = score[0].copy()
            vis = vis - vis.min()
            if vis.max() > 0:
                vis = vis / vis.max()
            vis = (vis * 255).astype(np.uint8)

            # Save with PIL
            im = Image.fromarray(vis)  # vis shape: (224,224)
            save_path = os.path.join(save_dir, f"anomaly.png")
            im.save(save_path)

        recon_imgs.extend(rgb.cpu().numpy())
        test_imgs.extend(ref.cpu().numpy())
        scores.append(score)

scores = np.asarray(scores).squeeze()
max_anomaly_score = scores.max()
min_anomaly_score = scores.min()
scores = (scores - min_anomaly_score) / (max_anomaly_score - min_anomaly_score)
gt_mask = np.concatenate([np.asarray(tf_mask(a))[None,...] for a in gt_masks], axis=0)

gt_mask = (gt_mask - gt_mask.min()) / (gt_mask.max() - gt_mask.min())
precision, recall, thresholds = precision_recall_curve(gt_mask.flatten(), scores.flatten())
a = 2 * precision * recall
b = precision + recall
f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
threshold = thresholds[np.argmax(f1)]

fpr, tpr, _ = roc_curve(gt_mask.flatten(), scores.flatten())
per_pixel_rocauc = roc_auc_score(gt_mask.flatten(), scores.flatten())
print('pixel ROCAUC: %.3f' % (per_pixel_rocauc))

au_pro, au_roc, pro_curve, roc_curve = calculate_au_pro_au_roc(gt_mask, scores)
print(f"aupro: {au_pro}. and other au_roc: {au_roc}")

img_scores = scores.reshape(scores.shape[0], -1).max(axis=1)
gt_list_isano = np.asarray(all_labels) != 0
img_roc_auc = roc_auc_score(gt_list_isano, img_scores)
print('image ROCAUC: %.3f' % (img_roc_auc))

print(f"avg_pose_time_ms  : {np.mean(times):.2f}")
print(f"avg_total_time_ms : {np.mean(total_times):.2f}")
print(f"total_time_ms : {np.sum(total_times):.2f}")

if args.use_wandb:
    wandb.log({
        "sum_total_time_ms": float(np.sum(total_times)),
        "avg_pose_time_ms": float(np.mean(times)),
        "avg_total_time_ms": float(np.mean(total_times)),
        "pixel_roc" : per_pixel_rocauc,
        "image_roc" : img_roc_auc,
        "aupro" : au_pro
    })