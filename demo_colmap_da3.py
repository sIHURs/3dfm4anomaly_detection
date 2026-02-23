import glob, os, torch
from factory.depth_anything_3.api import DepthAnything3
device = torch.device("cuda")
# choose model ckpt
model = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE")
model = model.to(device=device)
example_path = "/home/wangyifa/tmp/Depth-Anything-3/demo/01Gorilla_da_a6000/images"
images = sorted(glob.glob(os.path.join(example_path, "*.png")))

output_dir = "/home/wangyifa/tmp/Depth-Anything-3/demo/01Gorilla_da_a6000/outputs/colmap/sparse"
os.makedirs(output_dir, exist_ok=True)
export_format = "colmap"

prediction = model.inference(
    images,
    extrinsics=None,
    intrinsics=None,
    infer_gs=True,
    export_dir=output_dir,
    export_format=export_format,
    export_feat_layers=[],
)

# prediction.processed_images : [N, H, W, 3] uint8   array
print(prediction.processed_images.shape)
# prediction.depth            : [N, H, W]    float32 array
print(prediction.depth.shape)  
# prediction.conf             : [N, H, W]    float32 array
print(prediction.conf.shape)  
# prediction.extrinsics       : [N, 3, 4]    float32 array # opencv w2c or colmap format
print(prediction.extrinsics.shape)
# prediction.intrinsics       : [N, 3, 3]    float32 array
print(prediction.intrinsics.shape)