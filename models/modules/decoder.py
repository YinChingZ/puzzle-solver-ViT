import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureDecoder(nn.Module):
    """将特征图解码回原始图像空间的解码器"""
    
    def __init__(self, input_dim=256, output_dim=3, output_size=192):
        super().__init__()
        
        # 上采样因子 (从24x24到192x192需要8倍上采样)
        self.scale_factor = output_size // 24
        self.output_size = output_size
        
        # 解码器架构
        self.decoder = nn.Sequential(
            # 阶段1: 将特征维度从256减少到128
            nn.Conv2d(input_dim, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # 阶段2: 第一次上采样，24x24 -> 48x48
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # 阶段3: 第二次上采样，48x48 -> 96x96
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # 阶段4: 第三次上采样，96x96 -> 192x192
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            
            # 最终输出层: 16 -> 3通道
            nn.Conv2d(16, output_dim, kernel_size=3, padding=1),
            nn.Tanh()  # 将输出值映射到[-1,1]范围
        )
    
    def forward(self, x):
        """
        前向传播
        参数:
            x: 特征图 [B, C, H, W]
        返回:
            decoded: 重建图像 [B, 3, H*scale, W*scale]
        """
        # 如果输入是展平的特征（如ViT的输出），需要重新调整形状
        if len(x.shape) == 3:  # [B, N, D]
            batch_size, num_patches, dim = x.shape
            
            # 计算网格大小
            grid_size = int(num_patches ** 0.5)
            if grid_size ** 2 != num_patches:
                # 如果是PATCH_EMBEDDING，其中包含CLS TOKEN，需要移除CLS TOKEN
                grid_size = int((num_patches - 1) ** 0.5)
                if (grid_size ** 2) == (num_patches - 1):
                    # 移除CLS TOKEN
                    x = x[:, 1:, :]
                    num_patches = num_patches - 1
                    grid_size = int(num_patches ** 0.5)
            
            # 重新整形为[B, D, grid_size, grid_size]以用于卷积操作
            x = x.reshape(batch_size, grid_size, grid_size, dim).permute(0, 3, 1, 2)
            
        # 应用解码器网络
        decoded = self.decoder(x)
        
        # 确保输出尺寸正确
        if decoded.shape[-1] != self.output_size:
            decoded = F.interpolate(
                decoded, 
                size=(self.output_size, self.output_size),
                mode='bilinear', 
                align_corners=False
            )
        
        return decoded