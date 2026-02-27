
from argparse import ArgumentParser
import os
import wandb

import torch
import numpy as np
import random
from torchvision import transforms
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve
from scipy.ndimage import gaussian_filter
import torch.nn.functional as F

# needed for PAD code
from easydict import EasyDict
import yaml
from PIL import Image

from factory.splatpose.pose_estimation import main_pose_estimation
from factory.splatpose.utils_pose_est import ModelHelper, update_config, load_depth_outputs
from factory.splatpose.aupro import calculate_au_pro_au_roc

# solve path problem
from pathlib import Path
PAD_CONFIG_PATH = Path(__file__).resolve().parents[1] / "3dfm4anomaly_detection" / "factory" / "splatpose" /"PAD_utils" / "config_effnet.yaml"


MAD_classnames = ["01Gorilla", "02Unicorn", "03Mallard", "04Turtle", "05Whale", "06Bird", "07Owl", "08Sabertooth",
              "09Swan", "10Sheep", "11Pig", "12Zalika", "13Pheonix", "14Elephant", "15Parrot", "16Cat", "17Scorpion",
              "18Obesobeso", "19Bear", "20Puppy"]

PIAD_CL_classnames = ["Axletree", "Box", "Can", "Chain", "Gear", "Keyring", "Motor", "Parts", "Picker", "Section", "Shaft",
                      "Spray_can", "Spring", "Sprockets"]


DEBUG_CONF_FEATURE = False


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
pre_parser.add_argument("--gs_dir", type=str, help="3dgs output dir", default="output")
pre_parser.add_argument("--query_json_path", type=str, help="path of the query camera pose json file", default="query_json_path.json")
pre_parser.add_argument("--query_conf_map_path", type=str, help="path of the query images conf map", default="query_json_path.json")

args = pre_parser.parse_args()

if args.use_wandb:
    wandb.init(
        project="splatpose-anomaly",
        config=vars(args),
    )
else:
    wandb.init(mode="disabled")


def save_pose_estimation_outputs(
    model_dir,
    prefix,
    test_images,
    reference_images,
    all_labels,
    gt_masks,
    times,
    total_times,
    filenames,
):
    """
    Save pose estimation outputs (all lists) to model_dir as .npy files.
    """
    os.makedirs(model_dir, exist_ok=True)

    def _save(name, data):
        path = os.path.join(model_dir, f"{prefix}_{name}.npy")
        np.save(path, np.array(data, dtype=object), allow_pickle=True)
        print(f"[OK] Saved {path}")

    _save("test_images", test_images)
    _save("reference_images", reference_images)
    _save("all_labels", all_labels)
    _save("gt_masks", gt_masks)
    _save("times", times)
    _save("total_times", total_times)
    _save("filenames", filenames)

def load_pose_estimation_outputs(model_dir, prefix):
    """
    Load pose estimation outputs saved by save_pose_estimation_outputs.
    Returns all fields as Python lists.
    """
    def _load(name):
        path = os.path.join(model_dir, f"{prefix}_{name}.npy")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        return np.load(path, allow_pickle=True).tolist()

    test_images      = _load("test_images")
    reference_images = _load("reference_images")
    all_labels       = _load("all_labels")
    gt_masks         = _load("gt_masks")
    times            = _load("times")
    total_times      = _load("total_times")
    filenames        = _load("filenames")

    return (
        test_images,
        reference_images,
        all_labels,
        gt_masks,
        times,
        total_times,
        filenames,
    )


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

if DEBUG_CONF_FEATURE:
    _, depth_conf_query_images = load_depth_outputs(
        model_dir,
        prefix="vggt_query",
        as_torch=False,
        skip_first=10
    )

    depth_conf = depth_conf_query_images.astype(np.float32)

    min_v = depth_conf.min()
    max_v = depth_conf.max()

    depth_conf_norm = (depth_conf - min_v) / (max_v - min_v + 1e-8)

    # print("[DEBUG] depth_conf_query_images.shape: ", depth_conf_query_images.shape)

test_images, reference_images, all_labels, gt_masks, times, total_times, filenames = main_pose_estimation(cur_class=args.classname,
                                                                                    result_dir=result_dir,
                                                                                    model_dir_location=model_dir,
                                                                                    k=args.k, 
                                                                                    verbose=args.verbose,
                                                                                    data_dir=data_dir,
                                                                                    pcd_name=args.pcd_name,
                                                                                    json_name=args.json_name,
                                                                                    query_json_path=args.query_json_path,
                                                                                    loftr_batch=args.loftr_batch,
                                                                                    loftr_resolution=args.loftr_resolution,
                                                                                    retrieval=args.retrieval_model,
                                                                                    gs_dir=args.gs_dir)


## DEBUG ############################
# save_pose_estimation_outputs(
#     model_dir=model_dir,
#     prefix=args.classname,
#     test_images=test_images,
#     reference_images=reference_images,
#     all_labels=all_labels,
#     gt_masks=gt_masks,
#     times=times,
#     total_times=total_times,
#     filenames=filenames,
# )

# (
#     test_images,
#     reference_images,
#     all_labels,
#     gt_masks,
#     times,
#     total_times,
#     filenames,
# ) = load_pose_estimation_outputs(
#     model_dir=model_dir,
#     prefix=args.classname
# )

# print("test_images:", type(test_images), len(test_images), type(test_images[0]), len(test_images[0]))
# print("reference_images:", type(reference_images), len(reference_images))
# print("all_labels:", type(all_labels))
# print("gt_masks:", type(gt_masks))
# print("times:", type(times))
# print("total_times:", type(total_times))
# print("filenames:", type(filenames))

############################

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



### todo ###########################################################################
# confidence weighted anomaly score

if DEBUG_CONF_FEATURE:
    def conf_to_weight_np(C, c_low_percent=30, c_high_percent=90, gamma=2.0):
        """
        C: (H,W) float32 conf map
        return w in [0,1]
        """
        lo = np.percentile(C, c_low_percent)
        hi = np.percentile(C, c_high_percent)
        w = (C - lo) / (hi - lo + 1e-6)
        w = np.clip(w, 0.0, 1.0) ** gamma
        return w, (float(lo), float(hi))

    with torch.no_grad():
        for i in range(len(test_images)):
            if isinstance(test_images[i], list):
                test_images[i] = np.asarray(test_images[i])
            if isinstance(reference_images[i], list):
                reference_images[i] = np.asarray(reference_images[i])

            ref = tf_img(reference_images[i]).unsqueeze(0).cuda()
            rgb = tf_img(test_images[i]).unsqueeze(0).cuda()
            fileId = filenames[i]

            ref_feature = model(ref)
            rgb_feature = model(rgb)

            score = criterion(ref, rgb).sum(1, keepdim=True)
            for j in range(len(ref_feature)):
                s_act = ref_feature[j]
                mse_loss = criterion(s_act, rgb_feature[j]).sum(1, keepdim=True)
                score += F.interpolate(mse_loss, size=224, mode='bilinear', align_corners=False)

            # score: (B,1,224,224) -> numpy (B,224,224)
            score = score.squeeze(1).cpu().numpy().astype(np.float32)

            if args.verbose:
                save_dir = os.path.join(result_dir, "3dgs_imgs", fileId.split(".")[0])
                os.makedirs(save_dir, exist_ok=True)

                vis = score[0].copy()
                vis = vis - vis.min()
                if vis.max() > 0:
                    vis = vis / vis.max()
                vis = (vis * 255).astype(np.uint8)

                Image.fromarray(vis).save(os.path.join(save_dir, "anomaly.png"))

            # smooth anomaly map (keep your current behavior)
            # for b in range(score.shape[0]):
            #     score[b] = gaussian_filter(score[b], sigma=4)

            # =========================
            # NEW: soft conf suppression (global, no region split)
            # =========================
            conf_map_raw = depth_conf_norm[i]  # <-- TODO: 替换成你的 conf 来源

            # to numpy and resize to (224,224)
            if torch.is_tensor(conf_map_raw):
                c_t = conf_map_raw.detach().float()
                if c_t.ndim == 2:
                    c_t = c_t[None, None, ...]  # (1,1,Hc,Wc)
                elif c_t.ndim == 3:
                    c_t = c_t[None, ...]        # (1,1,Hc,Wc) if already (1,Hc,Wc)
                c_224 = F.interpolate(c_t, size=(224, 224), mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
            else:
                c_np = np.asarray(conf_map_raw, dtype=np.float32)
                c_t = torch.from_numpy(c_np)[None, None, ...]
                c_224 = F.interpolate(c_t, size=(224, 224), mode="bilinear", align_corners=False)[0, 0].numpy()

            # 可选：conf 平滑（一般不需要；需要的话 sigma=1~2）
            # c_224 = gaussian_filter(c_224, sigma=1)

            # conf -> weight
            w, (c_lo, c_hi) = conf_to_weight_np(c_224, c_low_percent=30, c_high_percent=90, gamma=1.0)

            # apply to all pixels
            lam = 0.7  # 惩罚强度：0~1，越大越抑制低conf

            for b in range(score.shape[0]):
                # ---- 2) 方案 C：先把 score 归一化到 0~1（每张独立）----
                s = score[b]
                s = s - s.min()
                s = s / (s.max() + 1e-8)

                # ---- 3) 方案 B：只惩罚低 conf（不奖励高 conf）----
                # w=1 -> factor=1
                # w=0 -> factor=1-lam
                factor = 1.0 - lam * (1.0 - w)

                score[b] = s * factor

            # =========================
            # save (unchanged except name)
            # =========================
            if args.verbose:
                save_dir = os.path.join(result_dir, "3dgs_imgs", fileId.split(".")[0])
                os.makedirs(save_dir, exist_ok=True)

                vis = score[0].copy()
                vis = vis - vis.min()
                if vis.max() > 0:
                    vis = vis / vis.max()
                vis = (vis * 255).astype(np.uint8)

                Image.fromarray(vis).save(os.path.join(save_dir, "anomaly_conf_soft.png"))

            recon_imgs.extend(rgb.cpu().numpy())
            test_imgs.extend(ref.cpu().numpy())
            scores.append(score)

if not DEBUG_CONF_FEATURE:
    ## orig #########################
    with torch.no_grad():
        # todo: use batch, not just single images at once
        for i in range(len(test_images)):
            ref=tf_img(reference_images[i]).unsqueeze(0).cuda()
            # print("[DEBUG] ref shape:", ref.shape)
            rgb=tf_img(test_images[i]).unsqueeze(0).cuda()
            # print("[DEBUG] rgb shape:", rgb.shape)
            fileId = filenames[i]
            # todo: torch.cat([ref, rgb], dim=0) then send into model, inference only once
            ref_feature=model(ref)
            # print("[DEBUG] ref_feature shape:", len(ref_feature), ref_feature[0].shape)
            rgb_feature=model(rgb)
            # print("[DEBUG] rgb_feature shape:", len(rgb_feature), rgb_feature[0].shape)
            score = criterion(ref, rgb).sum(1, keepdim=True)
            for i in range(len(ref_feature)):
                
                s_act = ref_feature[i]
                mse_loss = criterion(s_act, rgb_feature[i]).sum(1, keepdim=True)
                # print("[DEBUG] mse_loss.shape:", mse_loss.shape)
                score += torch.nn.functional.interpolate(mse_loss, size=score.shape[-2:], mode='bilinear', align_corners=False)

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

## todo ################################################
# Check the types of failed detections

# Generate predicted labels using the current threshold
# img_scores has already been computed above: shape (N,)
# threshold has also been computed above (based on maximizing pixel-level F1)
pred_labels = img_scores > threshold

# Find indices of misclassified samples
# gt_list_isano is a boolean array (True = ground-truth anomaly, False = ground-truth normal)

# Case A: False Positives (FP) - the sample is normal, but the model predicts anomaly
# Logic: predicted True AND ground truth False
fp_indices = np.where(pred_labels & ~gt_list_isano)[0]

# Case B: False Negatives (FN) - the sample is anomalous, but the model predicts normal
# Logic: predicted False AND ground truth True
fn_indices = np.where(~pred_labels & gt_list_isano)[0]

print("-" * 30)
print(f"Threshold used (pixel-level F1 max): {threshold:.4f}")
print(f"False positive indices: {fp_indices}")
print(f"False negative indices: {fn_indices}")
print("-" * 30)

print("False positive filenames:", [filenames[i] for i in fp_indices])
print("False negative filenames:", [filenames[i] for i in fn_indices])
#########################################################

if args.use_wandb:
    wandb.log({
        "sum_total_time_ms": float(np.sum(total_times)),
        "avg_pose_time_ms": float(np.mean(times)),
        "avg_total_time_ms": float(np.mean(total_times)),
        "pixel_roc" : per_pixel_rocauc,
        "image_roc" : img_roc_auc,
        "aupro" : au_pro
    })