import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastivePuzzleLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, features, positions):
        """
        计算对比损失，相同位置的特征应该更相似
        
        参数:
            features: 拼图块特征 [B, N, D] 或 [B, seq_len, D]
            positions: 真实位置标签 [B, N] 或 [B, N, K] 其中K是位置编码维度
        """
        # 处理class token（如果存在）
        if features.size(1) > 576:  # 假设576是最大拼图块数
            print(f"Removing class token from features. Original shape: {features.shape}")
            features = features[:, 1:, :]  # 移除class token
        
        batch_size, num_patches, dim = features.shape
        
        # L2归一化特征
        features = F.normalize(features, p=2, dim=2)
        
        # 计算特征间余弦相似度矩阵 [B, N, N]
        similarity = torch.bmm(features, features.transpose(1, 2)) / self.temperature
        
        # 创建位置相似度标签: 1表示同一位置，0表示不同位置
        position_similarity = torch.zeros_like(similarity)
        
        # 处理不同维度的positions
        for b in range(batch_size):
            for i in range(min(num_patches, positions.size(1))):
                for j in range(min(num_patches, positions.size(1))):
                    # 根据positions的维度选择比较方法
                    if positions.dim() == 3:  # [B, N, K]
                        # 比较整个位置向量是否相等
                        # 使用torch.all确保所有元素相等
                        if i < positions.size(1) and j < positions.size(1):
                            is_same = torch.all(positions[b, i] == positions[b, j]).item()
                            if is_same:
                                position_similarity[b, i, j] = 1.0
                    else:  # [B, N]
                        # 简单的标量比较
                        if positions[b, i].item() == positions[b, j].item():
                            position_similarity[b, i, j] = 1.0
        
        # 对角线上的元素(自身相似度)需要被忽略
        mask = torch.eye(num_patches, device=features.device)
        mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
        
        # 去除对角线
        similarity = similarity * (1 - mask)
        position_similarity = position_similarity * (1 - mask)
        
        # 计算正负样本损失
        pos_similarity = (similarity * position_similarity).sum(dim=2) + 1e-8  # 添加小值防止log(0)
        neg_similarity = (similarity * (1 - position_similarity)).sum(dim=2) + 1e-8
        
        # 最终对比损失
        loss = -torch.log(torch.exp(pos_similarity) / 
                         (torch.exp(pos_similarity) + torch.exp(neg_similarity)))
        
        return loss.mean()