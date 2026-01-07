# <Research: 3D Foundation Models for Multi-View Anomaly Detection>  
**proposal**: In the field of novel view synthesis and rendering of 3D scenes, 3D Gaussian Splatting (3DGS)
[1] has proven to be a groundbreaking new method. Initial works, such as SplatPose [2], have
already utilized this to detect anomalies in 3D objects using multi-view images.
New works on 3D foundation models have achieved increasingly stronger results without
training on the respective scenes. For example, the models VGGT [3] and MV-DUSt3R [4] use
a transformer architecture, whereas MVGD [5] pursues a similar strategy using a diffusion
model. All these models are trained on many different datasets and thus contain a great deal
of knowledge about three-dimensional structures and can derive these from a collection of
images without training.
In this work, combinations with these 3D foundation models are to be developed based on
SplatPose. These can either improve Gaussian Splatting or completely replace it. Possible
approaches for improved representation of the 3D object are to be investigated.
Finally, the foundation models are to be used for the task of anomaly detection in various
multi-view anomaly detection datasets and examined for their practicability and
effectiveness.

- [1] Kerbl et al. „3D Gaussian Splatting for Real-Time Radiance Field Rendering“
- [2] M. Kruse et al. „SplatPose & Detect: Pose-Agnostic 3D Anomaly Detection“
- [3] J. Wang et al. „VGGT: Visual Geometry Grounded Transformer“
- [4] Z. Tang et al. „MV-DUSt3R+: Single-Stage Scene Reconstruction from Sparse Views In 2 Seconds“
- [5] V. Guizilini et al. „Zero-Shot Novel View and Depth Synthesis with Multi-View Geometric Diffusion“

## 📌 Abstrcut
TODO

## 📦 Tested Methods
- vinilla gaussian splatting
- vggt-low-vram
- vggt-x
- anysplat
- instantSplat
- 3dgs mcmc
...
--

## 📰 Progress

**3dgs eval**





[ ] TODO

3dgs-mcmc:
In the update equations, additional random terms are introduced, which are applied only to the Gaussian xyz positions. This prevents the Gaussians from getting trapped in local optima in the spatial domain and allows some Gaussians to explore a wider region to find better solutions.

Furthermore, regarding densification control, the vanilla 3DGS strategy performs split and clone operations once the parameters of a Gaussian exceed certain thresholds. However, the initialization of the newly created Gaussians is fixed, and these new Gaussians may disrupt the previously reconstructed structure, thereby altering the original gradient optimization direction. In contrast, in the MCMC-based approach, after split and clone operations, the distribution of Gaussians is kept consistent with the original one.


**image-level ROCAUC(%) (Seed=0)**

|              | SplatPose | vggt_low+3dgs+optimize | vggt_low+3dgs+optimize (180views) | vggt_low+3dgs_mcmc |
|--------------|-----------|------------------------|-----------------------------------|--------------------|
| 01Gorilla    | 91.7±1.1  | 86.3 | 85.5 | 91.6 |
| 02Unicorn    | 97.9±1.1  | 85.1 | 84.9 | 97.5 |
| 03Mallard    | 97.4±0.5  | 75.5 | 83.5 | 96.6 |
| 04Turtle     | 97.2±0.7  | 62.3 | 76.1 | 96.8 |
| 05Whale      | 95.4±3.0  | 72.1 | 76.7 | 91.1 |
| 06Bird       | 94.0±1.2  | 79.6 | 86.6 | 88.6 |
| 07Owl        | 86.8±0.9  | 75.3 | 79.8 | 81.5 |
| 08Sabertooth | 95.2±1.5  | 55.8 | 65.7 | 85.0 |
| 09Swan       | 93.0±0.7  | 75.7 | 77.8 | 89.5 |
| 10Sheep      | 96.7±0.1  | 89.1 | 90.9 | 95.5 |
| 11Pig        | 96.1±1.9  | 87.9 | 89.4 | 94.5 |
| 12Zalika     | 89.9±0.7  | 83.6 | 86.0 | 90.8 |
| 13Pheonix    | 84.2±0.3  | 71.6 | 70.5 | 71.4 |
| 14Elephant   | 94.7±0.9  | 82.4 | 76.2 | 91.5 |
| 15Parrot     | 96.1±1.1  | 58.9 | 74.0 | 83.8 |
| 16Cat        | 84.2±1.3  | 82.6 | 81.7 | 87.8 |
| 17Scorpion   | 99.2±0.1  | 77.9 | 80.3 | 95.3 |
| 18Obesobeso  | 95.7±0.7  | 94.2 | 93.6 | 96.5 |
| 19Bear       | 98.9±0.2  | 83.7 | 86.8 | 96.7 |
| 20Puppy      | 96.1±0.9  | 77.4 | 78.4 | 90.8 |
| **mean**     | 93.9±0.2  | 77.9 | 81.2 | 90.6 |

Because of the limited evaluation time, all computations were performed only once, and the random seed was set to 0.

<!-- pixel-level ROCAUC

|              |  vggt+3dgs+optimize | Column C |
|--------------|---------------------|----------|
| 01Gorilla    | value 2             | value 3  |
| 02Unicorn    | value 5  | value 6  |
| 03Mallard    | value 2  | value 3  |
| 04Turtle     | value 5  | value 6  |
| 05Whale      | value 2  | value 3  |
| 06Bird       | value 5  | value 6  |
| 07Owl        | value 2  | value 3  |
| 08Sabertooth | value 5  | value 6  |
| 09Swan       | value 2  | value 3  |
| 10Sheep      | value 5  | value 6  |
| 11Pig        | value 2  | value 3  |
| 12Zalika     | value 5  | value 6  |
| 13Pheonix    | value 2  | value 3  |
| 14Elephant   | value 5  | value 6  |
| 15Parrot     | value 2  | value 3  |
| 16Cat        | value 5  | value 6  |
| 17Scorpion   | value 2  | value 3  |
| 18Obesobeso  | value 5  | value 6  |
| 19Bear       | value 2  | value 3  |
| 20Puppy      | value 5  | value 6  | -->

**SplatPose eval process improvement**
1. Move all loading-related code outside the loop to reduce the number of load operations.
2. Reduce unnecessary data transfers between the CPU and GPU.
3. Lower the LoFTR resolution (to 128) and rewrite the LoFTR retrieval to process images in batches instead of single images, with a batch size of 32.

| Setting | Avg. Pose Time (MM:SS) | Avg. Total Time (MM:SS) | Total Time (MM:SS) |
|--------|-------------------------|--------------------------|--------------------|
| Before Optimization | 00:03 | 00:09 | 40:12 |
| After Optimization  | 00:02 | 00:03 | 11:15 |

The above results correspond to the single-class evaluation on 01Gorilla. The `pose time` refers to the time spent on pose alignment, while the `total time` includes both the pose alignment time and the time required for LoFTR to search for a coarse initial pose.

todo:
- [ ] Sparse View 
    We evaluate the 3DFM reconstruction quality and the final anomaly detection performance under different input sparsity levels, using 80%, 60%, 40%, and 20% of the input images. Under sparse-view conditions, it becomes difficult to retrieve training images that are similar to the query images. To address this issue, we first obtain a coarse pose using VGGT, and then perform fine pose refinement by rotating the Gaussians.

    Under sparse-view Regularization, incorporating depth regularization leads to a more pronounced improvement in 3DGS performance. further integrate depth regularization with the 3DGS MCMC framework. (in process)

- [ ] Can we provide a visual explanation for why 3DGS MCMC performs well in our case?
- [ ] compare with other recent VGGT-derived models, including Pi3, FastVGGT, and VGGT-X...



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
# add the follow headers into the file (.h):
#include <cstdint>
#include <cstddef>

cd submodules/simple-knn
git checkout 86710c2
pip install . --no-build-isolation 

dc submodules/fused-ssim
git checkout 1272e21
pip install . --no-build-isolation 
```

### Framework Factory

3d foundation models:


Gaussian Splatting Models:

have some envs cuz diff_gaussian_rasterizatio version 