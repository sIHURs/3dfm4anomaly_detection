import matplotlib.pyplot as plt
import numpy as np

# 1. 硬编码数据 (请在此处填入你的实验结果)
k_values = [0, 25, 50, 75, 100, 125, 150, 175, 200]
inference_times = [0, 0.6, 1.2, 1.8, 2.4, 3.0, 3.6, 4.2, 4.8]  # 对应每个k的推断时间

# 指标数据 (示例数值，请根据实际情况修改)
image_auroc = [75.4, 82.6, 85.1, 87.3, 87.3, 87.2, 87.4, 87.2, 87.2]
pixel_auroc = [98.1, 98.3, 98.5, 98.6, 98.6, 98.6, 98.6, 98.6, 98.6]
pixel_aupro = [94.3, 94.9, 95.4, 95.4, 95.5, 95.5, 95.5, 95.5, 95.5]

# 2. 创建画布
fig, ax1 = plt.subplots(figsize=(8, 5), dpi=100)

# 3. 绘制主曲线 (以 inference time 为下方的 X 轴)
# 使用不同形状的标记: 'v' 倒三角, '^' 正三角, 's' 正方形
ax1.plot(inference_times, image_auroc, marker='v', label='image-auroc', linewidth=1.5)
ax1.plot(inference_times, pixel_auroc, marker='^', label='pixel-auroc', linewidth=1.5)
ax1.plot(inference_times, pixel_aupro, marker='s', label='(pixel-)aupro', linewidth=1.5)

# 4. 设置下方 X 轴和 Y 轴标签
ax1.set_xlabel('inference time (s)', fontsize=12)
ax1.set_ylabel('AUROC / AUPRO', fontsize=12)
ax1.grid(True, which='both', linestyle='-', alpha=0.7) # 开启网格

# 5. 创建上方的 X 轴 (用于显示 k 值)
ax2 = ax1.twiny() 
ax2.set_xlim(ax1.get_xlim()) # 确保两个轴的范围对齐
ax2.set_xticks(inference_times) # 在对应的推断时间位置打刻度
ax2.set_xticklabels([f'k = {k}' for k in k_values], rotation=45, ha='left') # 设置标签并旋转

# 6. 图例设置
ax1.legend(loc='lower right', frameon=True)

# 7. 细节微调
plt.tight_layout()

# 保存或显示
plt.savefig("overleaf/rad_ablation_k_plot_1.png", bbox_inches='tight')
# plt.show()