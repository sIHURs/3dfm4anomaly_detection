import os
import cv2
import numpy as np

def bg_to_transparent_hsv(input_dir, output_dir,
                          s_thresh=40, v_thresh=200):
    """
    使用 HSV 空间将白色/浅色背景变为透明。

    参数：
        input_dir  : 输入图片文件夹（支持 .jpg/.png）
        output_dir : 输出透明 PNG 的文件夹
        s_thresh   : 饱和度阈值，S < s_thresh 视为“背景候选”
        v_thresh   : 亮度阈值，V > v_thresh 视为“背景候选”
    """
    os.makedirs(output_dir, exist_ok=True)
    valid_ext = [".png", ".jpg", ".jpeg"]

    for fname in sorted(os.listdir(input_dir)):
        if not any(fname.lower().endswith(ext) for ext in valid_ext):
            continue

        in_path = os.path.join(input_dir, fname)
        img_bgr = cv2.imread(in_path)
        if img_bgr is None:
            print(f"读取失败，跳过: {in_path}")
            continue

        # 1) BGR -> HSV
        img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(img_hsv)

        # 2) 找到“高亮 + 低饱和”的背景区域
        #    背景条件：S < s_thresh 且 V > v_thresh
        bg_mask = np.logical_and(s < s_thresh, v > v_thresh)

        # 3) 构造 alpha 通道：背景=0（透明），前景=255（不透明）
        alpha = np.zeros_like(v, dtype=np.uint8)
        alpha[~bg_mask] = 255

        # 4) 合成 BGRA 图像
        img_bgra = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2BGRA)
        img_bgra[:, :, 3] = alpha

        # 5) 输出为 PNG
        out_name = os.path.splitext(fname)[0] + ".png"
        out_path = os.path.join(output_dir, out_name)
        cv2.imwrite(out_path, img_bgra)

        print(f"处理完成: {fname} -> {out_name}")

    print("全部图片转换完成！")


# 示例用法
input_folder = "scripts/demo_PIAD_real_vggt_3dgs/01Valve/images_bg"              # 你的自定义 images 文件夹
output_folder = "scripts/demo_PIAD_real_vggt_3dgs/01Valve/images" # 输出透明结果

bg_to_transparent_hsv(input_folder, output_folder,
                      s_thresh=40, v_thresh=200)
