import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from models.modules.decoder import FeatureDecoder

class ReconstructionModule(nn.Module):
    def __init__(self, input_dim=768, output_dim=3, grid_size=4, img_size=192):
        super().__init__()
        self.grid_size = grid_size
        self.img_size = img_size
        
        # 现有的重建逻辑...
        
        # 添加解码器将特征图转换为图像
        self.decoder = FeatureDecoder(
            input_dim=input_dim, 
            output_dim=output_dim, 
            output_size=img_size
        )
    
    def forward(self, features, position_indices, relation_logits=None):
        """
        根据预测的位置和关系重建图像
        """
        # 从输入特征中检测是否存在class token
        batch_size, seq_len, dim = features.shape
        
        # 如果存在class token，移除它
        if seq_len > self.grid_size * self.grid_size:
            print(f"Detected class token, removing it. Original shape: {features.shape}")
            features = features[:, 1:, :]
            print(f"Reshaping to: [{batch_size}, {int(np.sqrt(features.size(1)))}, {int(np.sqrt(features.size(1)))}, {features.size(2)}], num_patches={features.size(1)}")
        
        # 将特征重新组织成网格形式
        grid_features = features.reshape(batch_size, -1, dim)
        
        # 根据位置索引重排拼图块
        confidence = torch.ones(batch_size, device=features.device)
        
        # 使用解码器将特征解码为图像
        reconstructed_image = self.decoder(grid_features)
        
        return reconstructed_image, confidence
