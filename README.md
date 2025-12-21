# <Project Title>  
Official Code for **"<Paper Title>"**  
<Conference / Journal, Year>  
[📄 Paper](#) | [🌐 Project Page](#) | [📦 Dataset](#) | [🎥 Video](#)

---

## 📌 Overview  
This repository contains the implementation of the Master Thesis @yifan:

Our method proposes **<short summary of the contribution>**, which enables:
- ✔️ <Key point 1>  
- ✔️ <Key point 2>  
- ✔️ <Key point 3>  

---

## 📦 Contained Works
- vinilla gaussian splatting
- vggt-low-vram
- vggt-x

todo:
- [ ] splatpose 
- [ ] splatpose++
- other 3dgs model

## 📰 Progress
- **[2024-11]** Initial release of the repository.  
- **[2024-12]** Added training & evaluation scripts.  
- **[2025-01]** Pre-trained models released.  

---

## 📦 Environment Setup  

### 1. Create Conda Environment
```bash
# create own conda-env
conda create -n -f <env_name> python==3.11 -y
conda activate <env_name>

# install colmap
conda install -c conda-forge colmap

# the main task run on rtx3090 24G - with env torch 2.8.0 cuda 12.9 toolkit
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu129

# other depencies
pip install --no-build-isolation -r requirements.txt

# locally build

cd submodules/diff-gaussian-rasterization
git checkout 9c5c202
pip install  . --no-build-isolation 

# Optional: if use torch2.8.0 + cuda12.9 and turing arch, add header in rasterizer_impl.h

cd ~/tmp/3dfm4anomaly_detection/submodules/diff-gaussian-rasterizatio
vim cuda_rasterizer/rasterizer_impl.h
# add the follow headers into the file
# #include <cstdint>
# #include <cstddef>

cd submodules/simple-knn
git checkout 86710c2
pip install . --no-build-isolation 

dc submodules/fused-ssim
git checkout 1272e21
pip install . --no-build-isolation 
