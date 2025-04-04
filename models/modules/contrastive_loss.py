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
            features: 拼图块特征 [B, N, D]
            positions: 真实位置标签 [B, N]
        """
        batch_size, num_patches, dim = features.shape
        
        # L2归一化特征
        features = F.normalize(features, p=2, dim=2)
        
        # 计算特征间余弦相似度矩阵 [B, N, N]
        similarity = torch.bmm(features, features.transpose(1, 2)) / self.temperature
        
        # 创建位置相似度标签: 1表示同一位置，0表示不同位置
        position_similarity = torch.zeros_like(similarity)
        for b in range(batch_size):
            for i in range(num_patches):
                for j in range(num_patches):
                    # 相同位置或相邻位置设为正样本
                    if positions[b, i] == positions[b, j]:
                        position_similarity[b, i, j] = 1.0
        
        # 对角线上的元素(自身相似度)需要被忽略
        mask = torch.eye(num_patches).to(features.device)
        mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
        
        # 去除对角线
        similarity = similarity * (1 - mask)
        position_similarity = position_similarity * (1 - mask)
        
        # 计算正负样本损失
        pos_similarity = (similarity * position_similarity).sum(dim=2)
        neg_similarity = (similarity * (1 - position_similarity)).sum(dim=2)
        
        # 最终对比损失
        loss = -torch.log(torch.exp(pos_similarity) / 
                          (torch.exp(pos_similarity) + torch.exp(neg_similarity) + 1e-8))
        
        return loss.mean()