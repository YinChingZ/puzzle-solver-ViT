import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

class PatchGenerator:
    def __init__(self, grid_size=4, patch_size=None):
        """
        初始化拼图块生成器
        
        参数:
            grid_size: 网格的行/列数
            patch_size: 每个块的大小（如果不指定，将根据图像尺寸自动计算）
        """
        self.grid_size = grid_size
        self.patch_size = patch_size
    
    def generate_patches(self, image, return_positions=True):
        """
        从图像生成拼图块
        
        参数:
            image: PIL Image或torch.Tensor格式的图像
            return_positions: 是否返回位置信息
            
        返回:
            patches: 拼图块张量
            positions: (如果return_positions=True) 位置索引
        """
        # 1. 检查输入类型，必要时进行转换
        if isinstance(image, Image.Image):
            image = transforms.ToTensor()(image).unsqueeze(0)
        elif isinstance(image, torch.Tensor):
            if image.dim() == 3:
                image = image.unsqueeze(0)
        else:
            raise TypeError("Unsupported image type. Expected PIL Image or torch.Tensor.")
        
        batch_size, channels, height, width = image.size()
        
        # 2. 计算块尺寸
        if self.patch_size is None:
            self.patch_size = min(height, width) // self.grid_size
        
        patches = []
        positions = []
        
        # 3. 分割图像
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                patch = image[:, :, i * self.patch_size:(i + 1) * self.patch_size, j * self.patch_size:(j + 1) * self.patch_size]
                patches.append(patch)
                if return_positions:
                    positions.append((i, j))
        
        patches = torch.cat(patches, dim=0)
        
        # 4. 生成位置索引
        if return_positions:
            positions = torch.tensor(positions)
            return patches, positions
        
        # 5. 返回结果
        return patches
