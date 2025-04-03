import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionHead(nn.Module):
    def __init__(self, input_dim, num_positions):
        super(PositionHead, self).__init__()
        self.fc = nn.Linear(input_dim, num_positions)
        self.num_positions = num_positions

    def forward(self, x, return_indices=False):
        logits = self.fc(x)
        
        if not return_indices:
            # 默认返回原始 logits（浮点类型）用于损失计算
            return logits
        
        # 如果需要索引（用于重建或评估），则计算并返回索引
        # 对每个位置的 logits 应用 softmax
        probs = F.softmax(logits, dim=-1)
        
        # 获取最大概率的索引
        position_indices = torch.argmax(probs, dim=-1)
        
        # 确保索引在有效范围内
        position_indices = torch.clamp(position_indices, 0, self.num_positions - 1)
        
        return position_indices