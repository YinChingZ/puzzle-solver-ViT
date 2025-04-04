import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureDecoder(nn.Module):
    """将高维特征图解码为RGB图像的解码器"""
    def __init__(self, in_channels=192, out_channels=3):
        super(FeatureDecoder, self).__init__()
        
        # 解码器层
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        
        self.conv3 = nn.Conv2d(32, out_channels, kernel_size=1)
        
        # 上采样 - 如果需要从24x24放大到更高分辨率
        self.upsample = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.conv3(x)
        x = torch.tanh(x)  # 输出范围为[-1,1]
        
        # 如果需要上采样到更大尺寸
        x = self.upsample(x)
        
        return x