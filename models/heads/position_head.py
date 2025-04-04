import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionHead(nn.Module):
    def __init__(self, input_dim, num_positions):
        super(PositionHead, self).__init__()
        self.fc = nn.Linear(input_dim, num_positions)
        self.num_positions = num_positions

    def forward(self, x, return_indices=False):
        # 输入 x 的形状是 [batch_size, seq_len, embed_dim]
        batch_size, seq_len, _ = x.shape
        
        # 生成 logits [batch_size, seq_len, num_positions]
        logits = self.fc(x)
        
        if return_indices:
            # 获取每个位置的最可能的索引
            indices = torch.argmax(logits, dim=-1)  # [batch_size, seq_len]
            return indices
        
        # 默认返回原始 logits 用于损失计算
        return logits  # [batch_size, seq_len, num_positions]
