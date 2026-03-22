# # 克隆官方仓库
# git clone https://github.com/hustvl/EVF-SAM.git
# cd EVF-SAM

# # 创建并激活环境
# conda create -n evfsam python=3.10 -y
# conda activate evfsam

# # 安装基础依赖 (根据你的 CUDA 版本选择)
# pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
# pip install -r requirements.txt
# pip install opencv-python pillow

import torch
import cv2
import numpy as np
from PIL import Image
import argparse
import os

# 注意：模型尚未支持 AutoModel，需从源码导入
from model.segment_anything.utils.transforms import ResizeLongestSide
from model.evf_sam2 import EvfSam2Model
from transformers import AutoTokenizer

def run_inference(image_path, prompt, model_path, output_path="result.jpg"):
    # 1. 加载模型
    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right", use_fast=False)
    
    # 根据模型路径加载对应的 EVF-SAM2
    model = EvfSam2Model.from_pretrained(model_path, torch_dtype=torch.float16).cuda()
    model.eval()

    # 2. 图像预处理
    image_np = cv2.imread(image_path)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    original_size = image_np.shape[:2]
    
    # 简单的 Prompt 处理
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].cuda()

    # 3. 模型推理
    print(f"Inferring with prompt: '{prompt}'...")
    with torch.no_grad():
        # EVF-SAM2 接受图像和文本输入进行早期融合
        output = model.generate(
            image_np, 
            input_ids, 
            precision='fp16'
        )
        # 获取分割掩码 (Mask)
        mask = output["mask"]

    # 4. 可视化并保存
    # 将掩码叠加到原图上
    mask_visual = (mask.cpu().numpy() * 255).astype(np.uint8)
    res_image = image_np.copy()
    res_image[mask_visual > 0] = res_image[mask_visual > 0] * 0.5 + np.array([0, 255, 0]) * 0.5
    
    cv2.imwrite(output_path, cv2.cvtColor(res_image.astype(np.uint8), cv2.COLOR_RGB2BGR))
    print(f"Result saved to {output_path}")

if __name__ == "__main__":
    # 配置参数
    IMG_PATH = "assets/zebra.jpg" # 替换为你的图片路径
    TEXT_PROMPT = "zebra on the left" # 你的文本描述
    MODEL_ID = "YxZhang/evf-sam2" # HuggingFace 上的模型 ID

    run_inference(IMG_PATH, TEXT_PROMPT, MODEL_ID)