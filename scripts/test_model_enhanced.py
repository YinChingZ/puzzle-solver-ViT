import os
import sys
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.puzzle_solver import PuzzleSolver

# 配置和模型加载代码...

def visualize_multi_channel(tensor, output_path, title="特征图可视化"):
    """创建多种不同的高维特征图可视化"""
    if tensor.device.type != 'cpu':
        tensor = tensor.cpu()
    
    if len(tensor.shape) == 4:
        tensor = tensor[0]  # 取第一个样本
    
    C, H, W = tensor.shape
    
    # 创建4种不同的可视化
    plt.figure(figsize=(20, 15))
    
    # 1. 前3个通道作为RGB
    plt.subplot(2, 2, 1)
    if C >= 3:
        img1 = tensor[:3].permute(1, 2, 0).numpy()
    else:
        channels = [tensor[i] for i in range(C)]
        while len(channels) < 3:
            channels.append(channels[-1])
        img1 = torch.stack(channels).permute(1, 2, 0).numpy()
    
    img1 = (img1 - img1.min()) / (img1.max() - img1.min() + 1e-6)
    plt.imshow(img1)
    plt.title("前3个通道作为RGB")
    plt.axis('off')
    
    # 2. 平均每64个通道
    plt.subplot(2, 2, 2)
    num_groups = min(3, C // 1)
    grouped_channels = []
    
    for i in range(num_groups):
        start_idx = i * (C // num_groups)
        end_idx = (i + 1) * (C // num_groups) if i < num_groups - 1 else C
        grouped_channels.append(tensor[start_idx:end_idx].mean(dim=0))
    
    while len(grouped_channels) < 3:
        grouped_channels.append(grouped_channels[-1])
    
    img2 = torch.stack(grouped_channels).permute(1, 2, 0).numpy()
    img2 = (img2 - img2.min()) / (img2.max() - img2.min() + 1e-6)
    plt.imshow(img2)
    plt.title("通道组平均")
    plt.axis('off')
    
    # 3. 使用通道的均值、标准差和最大值
    plt.subplot(2, 2, 3)
    mean_channel = tensor.mean(dim=0, keepdim=True)
    std_channel = tensor.std(dim=0, keepdim=True)
    max_channel = tensor.max(dim=0, keepdim=True)[0]
    
    stat_channels = torch.cat([mean_channel, std_channel, max_channel], dim=0)
    img3 = stat_channels.permute(1, 2, 0).numpy()
    img3 = (img3 - img3.min()) / (img3.max() - img3.min() + 1e-6)
    plt.imshow(img3)
    plt.title("均值/标准差/最大值")
    plt.axis('off')
    
    # 4. 显示重要特征通道
    plt.subplot(2, 2, 4)
    # 计算每个通道的方差，选择方差最大的3个通道（包含最多信息）
    var_per_channel = tensor.var(dim=(1, 2))
    top_channels = torch.topk(var_per_channel, min(3, C)).indices
    
    feature_channels = []
    for idx in top_channels:
        feature_channels.append(tensor[idx])
    
    while len(feature_channels) < 3:
        feature_channels.append(feature_channels[-1])
    
    img4 = torch.stack(feature_channels).permute(1, 2, 0).numpy()
    img4 = (img4 - img4.min()) / (img4.max() - img4.min() + 1e-6)
    plt.imshow(img4)
    plt.title("高方差通道")
    plt.axis('off')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"多通道可视化已保存到 {output_path}")

# 在主脚本中
# 模型推理后...

# 保存基本的比较图
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
visualize_tensor(original_tensor[0], title="原始图像")
plt.subplot(1, 3, 2)
visualize_tensor(shuffled_image[0], title="打乱的图像")
plt.subplot(1, 3, 3)
visualize_tensor(reconstructed_image[0], title="重建图像 (基本)", normalize=False)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'basic_comparison.png'), dpi=300)

# 保存高级特征图可视化
visualize_multi_channel(
    reconstructed_image[0], 
    os.path.join(output_dir, 'advanced_visualization.png'),
    title="重建特征的多种可视化方式"
)

print("可视化完成，结果已保存")