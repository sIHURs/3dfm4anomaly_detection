#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import shutil
import random

import glob
import os, re
import copy
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import trimesh
import pycolmap
import cv2
import json
from tqdm import tqdm
from datetime import datetime
import math
now = datetime.now()
from utils.time_recorder import SpanTimer

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve
from scipy.ndimage import gaussian_filter

from PIL import Image
from easydict import EasyDict
import yaml

from factory.splatpose.pose_estimation import render_from_estimated_poses
from factory.splatpose.utils_pose_est import ModelHelper, update_config, load_depth_outputs
from factory.splatpose.aupro import calculate_au_pro_au_roc

# Configure CUDA settings
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

# from factory.vggt_low_vram.vggt.models.vggt import VGGT
# from factory.vggt_low_vram.vggt.utils.load_fn import load_and_preprocess_images_square
# from factory.vggt_low_vram.vggt.utils.pose_enc import pose_encoding_to_extri_intri
# from factory.vggt_low_vram.vggt.utils.geometry import unproject_depth_map_to_point_map
# from factory.vggt_low_vram.vggt.utils.helper import create_pixel_coordinate_grid, randomly_limit_trues
# from factory.vggt_low_vram.vggt.dependency.track_predict import predict_tracks
# from factory.vggt_low_vram.vggt.dependency.np_to_pycolmap import batch_np_matrix_to_pycolmap, batch_np_matrix_to_pycolmap_wo_track




#### DEBUG FLAGS
DEBUG_CREATE_SAMPLE_MATCHING_JSON = False

#### code constants
DIGIT_RE = re.compile(r"(\d+)")
QUERY_CLASS_RE = re.compile(r"^query([A-Za-z]+)_")


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


def vprint(msg: str, verbose: bool):
    if verbose:
        print(msg)


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def safe_clean_dir(out_root: Path, verbose: bool, do_clean: bool):
    if do_clean and out_root.exists():
        vprint(f"[CLEAN] remove {out_root}", verbose)
        shutil.rmtree(out_root)


def place_file(src: Path, dst: Path, mode: str):
    ensure_dir(dst.parent)
    if dst.exists():
        return

    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "symlink":
        # Prefer relative symlink for portability
        try:
            rel = os.path.relpath(src, start=dst.parent)
            dst.symlink_to(rel)
        except Exception:
            dst.symlink_to(src)
    elif mode == "hardlink":
        os.link(src, dst)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def candidate_names(name: str, pad: int = 3, default_ext: str = ".png") -> List[str]:
    """
    Create filename candidates to handle:
      - querygood_000.png -> 000.png
      - 1.png -> 001.png
    Order matters: first try original, then digit-based.
    """
    ext = Path(name).suffix or default_ext
    cands = [name if Path(name).suffix else (name + ext)]

    digits = DIGIT_RE.findall(name)
    if digits:
        num = digits[-1]  # use last digit group
        cands.append(f"{num}{ext}")            # maybe already padded
        cands.append(f"{num.zfill(pad)}{ext}") # enforce padding

    # de-dup, keep order
    seen = set()
    out = []
    for c in cands:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


def infer_query_subdir(query_key: str) -> Optional[str]:
    """
    querymissing_003.png -> 'missing'
    querygood_008.png    -> 'good'
    """
    m = QUERY_CLASS_RE.match(query_key)
    if not m:
        return None
    return m.group(1).lower()


def find_image_by_candidates(
    name: str,
    root: Path,
    pad: int = 3,
    verbose: bool = False,
) -> Optional[Path]:
    """
    Try multiple candidate filenames under a single root directory.
    """
    for cand in candidate_names(name, pad=pad):
        p = root / cand
        if p.exists():
            return p
    if verbose:
        vprint(f"[DEBUG] not found under {root}: {name} cands={candidate_names(name, pad)}", verbose)
    return None


def parse_matching(matching_json: Path) -> Tuple[dict, Dict[str, List[str]]]:
    with matching_json.open("r", encoding="utf-8") as f:
        data = json.load(f)

    matches = data.get("matches", {})
    selection: Dict[str, List[str]] = {}
    for qname, lst in matches.items():
        if not isinstance(lst, list):
            continue
        train_names = []
        for item in lst:
            if isinstance(item, dict) and "train_file_path" in item:
                train_names.append(item["train_file_path"])
        selection[qname] = train_names
    return data, selection


def main():

    # prepare machted image sets for VGGT and 3DGS
    ap = argparse.ArgumentParser(
        description="Extract per-query (query + topk train) image sets from matching.json for VGGT and 3DGS (clean query-subdir mapping)."
    )
    ap.add_argument("--matching_json", required=True, type=str)
    ap.add_argument("--data_root", required=True, type=str, help="e.g. data/Anomaly_refine_msk")
    ap.add_argument("--out_root", required=True, type=str)

    ap.add_argument("--topk", type=int, default=None, help="Override topk; otherwise use json 'topk' or list as-is.")
    ap.add_argument("--pad", type=int, default=3, help="Zero-pad width for numeric filenames, default 3 -> 001.png")

    ap.add_argument("--mode", choices=["copy", "symlink", "hardlink"], default="copy")

    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--clean", action="store_true",
                    help="Force clean out_root before writing. Default: clean when NOT verbose; keep when verbose.")
    
    # splatpose pipeline
    ap.add_argument("-k", metavar="K", type=int, help="number of pose estimation steps", default=175)
    ap.add_argument("-c", "--classname", metavar="c", type=str, help="current class to run experiments on",
                            default="01Gorilla")
    ap.add_argument("-w", "--use_wandb", type=int, help="the wandb to use", default=0)
    ap.add_argument("-p", "--prefix", metavar="pf", type=str, help="prefix for the wandb run name", default="to_delete")
    ap.add_argument("--seed", type=int, help="seed for random behavior", default=0)
    ap.add_argument("--loftr_batch", type=int, help="batch size for loftr pose retrieval", default=32)
    ap.add_argument("--loftr_resolution", type=tuple, help="images resolution for loftr pose retrieval", default=(128,128))
    ap.add_argument("--gauss_iters", type=int, help="number of training iterations for 3DGS", default=30000)
    ap.add_argument("--wandb", type=int, help="whether we track with wandb", default=1)
    # ap.add_argument("--train", type=int, help="whether we train or look for a saved model", default=1)               
    ap.add_argument("-v", "--verbose", type=int, help="verbosity", default=0)                        
    ap.add_argument("--data_path", type=str, help="path pointing towards the usable data set", default="MAD-Sim_3dgs/")                        
    ap.add_argument("--result", type=str, help="path of output result", default="ad_result")
    ap.add_argument("--model_path_splatpose", type=str, help="path of 3dgs output model", default="output")
    ap.add_argument("--pcd_name", type=str, help="name of the processed 3dgs poind cloud", default="point_cloud.ply")
    ap.add_argument("--json_name", type=str, help="name of the camera pose json file", default="transforms.json")
    ap.add_argument("--retrieval_model", type=str, help="model for init c2w", default="loftr")
    ap.add_argument("--query_json_path", type=str, help="path of the query camera pose json file", default="query_json_path.json")
    ap.add_argument("--query_conf_map_path", type=str, help="path of the query images conf map", default="query_json_path.json")

    # create args
    args = ap.parse_args()

    # set seed
    seed_everything(args.seed, deterministic=True)

    if DEBUG_CREATE_SAMPLE_MATCHING_JSON:
        matching_json = Path(args.matching_json).expanduser().resolve()
        data_root = Path(args.data_root).expanduser().resolve()
        out_root = Path(args.out_root).expanduser().resolve()

        cfg, selection = parse_matching(matching_json)

        json_topk = cfg.get("topk")
        topk = args.topk if args.topk is not None else (json_topk if isinstance(json_topk, int) else None)

        # Fixed dataset structure
        train_root = data_root / "train" / "good"
        test_root = data_root / "test"  # contains subfolders: good/, missing/, stains/, ...

        if not train_root.exists():
            raise FileNotFoundError(f"train_root not found: {train_root}")
        if not test_root.exists():
            raise FileNotFoundError(f"test_root not found: {test_root}")

        # cleaning policy
        do_clean = args.clean or (not args.verbose)
        safe_clean_dir(out_root, verbose=args.verbose, do_clean=do_clean)
        ensure_dir(out_root)

        vggt_base = out_root / "vggt_sets"
        gs_base = out_root / "gs_sets"
        ensure_dir(vggt_base)
        ensure_dir(gs_base)

        missing: List[str] = []
        manifest_all = {
            "matching_json": str(matching_json),
            "data_root": str(data_root),
            "train_root": str(train_root),
            "test_root": str(test_root),
            "topk_used": topk,
            "pad": args.pad,
            "mode": args.mode,
            "items": {},
        }

        for query_key, train_list in selection.items():
            if topk is not None:
                train_list = train_list[:topk]

            # infer query subdir from name
            subdir = infer_query_subdir(query_key)
            if subdir is None:
                # fallback: assume 'good'
                subdir = "good"
            query_root = test_root / subdir

            vprint(f"\n[QUERY] {query_key} (subdir={subdir}) -> {train_list}", args.verbose)

            if not query_root.exists():
                missing.append(f"query dir not found for {query_key}: {query_root}")
                vprint(f"[WARN] query dir missing: {query_root}", args.verbose)
                continue

            q_path = find_image_by_candidates(query_key, query_root, pad=args.pad, verbose=args.verbose)
            if q_path is None:
                missing.append(
                    f"query not found: {query_key} under {query_root} (cands={candidate_names(query_key, args.pad)})"
                )
                vprint(f"[WARN] query missing: {query_key}", args.verbose)
                continue

            train_paths: List[Path] = []
            for tn in train_list:
                t_path = find_image_by_candidates(tn, train_root, pad=args.pad, verbose=args.verbose)
                if t_path is None:
                    missing.append(
                        f"train not found for {query_key}: {tn} under {train_root} (cands={candidate_names(tn, args.pad)})"
                    )
                    vprint(f"[WARN] train missing: {tn} (query {query_key})", args.verbose)
                    continue
                train_paths.append(t_path)

            # output dirs
            vggt_dir = vggt_base / query_key / "images"
            gs_train_dir = gs_base / query_key / "images" / "train"
            gs_query_dir = gs_base / query_key / "images" / "query"
            ensure_dir(vggt_dir)
            ensure_dir(gs_train_dir)
            ensure_dir(gs_query_dir)

            query_dst_name = query_key if Path(query_key).suffix else (query_key + ".png")
            # VGGT set: all images together (train + query)
            for p in train_paths:
                place_file(p, vggt_dir / p.name, args.mode)
            place_file(q_path, vggt_dir / query_dst_name, args.mode)            


            # 3DGS set: split train/query
            for p in train_paths:
                place_file(p, gs_train_dir / p.name, args.mode)
            place_file(q_path, gs_query_dir / query_dst_name, args.mode)

            per = {
                "query_key": query_key,
                "query_subdir": subdir,
                "query_src": str(q_path),
                "train_keys": train_list,
                "train_src": [str(p) for p in train_paths],
                "outputs": {
                    "vggt_images_dir": str(vggt_dir),
                    "gs_train_dir": str(gs_train_dir),
                    "gs_query_dir": str(gs_query_dir),
                },
            }
            (gs_base / query_key / "manifest.json").write_text(json.dumps(per, indent=2), encoding="utf-8")

            manifest_all["items"][query_key] = per

        (out_root / "manifest_all.json").write_text(json.dumps(manifest_all, indent=2), encoding="utf-8")

        if missing:
            (out_root / "missing.txt").write_text("\n".join(missing) + "\n", encoding="utf-8")
            print(f"[DONE] Output: {out_root}")
            print(f"[WARN] Missing: {len(missing)} -> {out_root/'missing.txt'}")
        else:
            print(f"[DONE] Output: {out_root}")
            print("[OK] All files found.")

    # todo: integrate the 3dgs training into the main_pose_estimation, careful about storage management
    #################### ad pipeline  #########################

    result_dir = os.path.join(args.result, f"results_{args.prefix}_{args.seed}", args.classname)
    model_dir = os.path.join(args.model_path_splatpose, args.classname)
    data_dir = args.data_path

    test_images, reference_images, all_labels, gt_masks, times, total_times, filenames = render_from_estimated_poses(cur_class=args.classname,
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

                                                                       retrieval=args.retrieval_model)
    PAD_CONFIG_PATH = Path(__file__).resolve().parents[1] / "3dfm4anomaly_detection" / "factory" / "splatpose" /"PAD_utils" / "config_effnet.yaml"
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
            ref_feature=model(ref)
            rgb_feature=model(rgb)
            # print("[DEBUG] ref shape:", ref.shape)
            # print("[DEBUG] rgb shape:", rgb.shape)
            # print("[DEBUG] ref_feature shape:", len(ref_feature), ref_feature[0].shape)
            # print("[DEBUG] rgb_feature shape:", len(rgb_feature), rgb_feature[0].shape)
            score = criterion(ref, rgb).sum(1, keepdim=True)
            for i in range(len(ref_feature)):
                
                s_act = ref_feature[i]
                mse_loss = criterion(s_act, rgb_feature[i]).sum(1, keepdim=True)
                # print("[DEBUG] mse_loss.shape:", mse_loss.shape)
                score += torch.nn.functional.interpolate(mse_loss, size=score.shape[-2:], mode='bilinear', align_corners=False)

            score = score.squeeze(1).cpu().numpy()
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

if __name__ == "__main__":
    main()