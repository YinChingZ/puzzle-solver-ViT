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
        # 转换输入为张量
        if isinstance(image, str):  # 文件路径
            image = Image.open(image).convert('RGB')
            transform = transforms.ToTensor()
            image = transform(image)
        elif isinstance(image, Image.Image):  # PIL图像
            transform = transforms.ToTensor()
            image = transform(image)
        elif isinstance(image, np.ndarray):  # NumPy数组
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        elif isinstance(image, torch.Tensor):
            # 确保图像格式为[C, H, W]或[B, C, H, W]
            if image.dim() == 4:  # 批次格式
                pass
            elif image.dim() == 3:  # 单图像格式
                image = image.unsqueeze(0)  # 添加批次维度
            else:
                raise ValueError(f"Unexpected tensor dimensions: {image.shape}")
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")
        
        # 获取批次大小和图像尺寸
        batch_size, channels, height, width = image.shape
        
        # 计算块大小
        patch_h = height // self.grid_size
        patch_w = width // self.grid_size
        
        # 创建拼图块和位置索引
        patches = []
        positions = []
        
        for b in range(batch_size):
            batch_patches = []
            batch_positions = []
            
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    # 提取拼图块
                    y_start = i * patch_h
                    y_end = (i + 1) * patch_h
                    x_start = j * patch_w
                    x_end = (j + 1) * patch_w
                    
                    patch = image[b, :, y_start:y_end, x_start:x_end]
                    batch_patches.append(patch)
                    
                    # 记录位置索引
                    pos = i * self.grid_size + j
                    batch_positions.append(pos)
            
            patches.append(torch.stack(batch_patches))
            positions.append(torch.tensor(batch_positions))
        
        # 组合批次中的所有拼图块和位置
        patches = torch.stack(patches)  # [B, N, C, H', W']
        positions = torch.stack(positions)  # [B, N]
        
        if return_positions:
            return patches, positions
        return patches
