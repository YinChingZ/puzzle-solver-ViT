import torch
import torch.nn as nn
import torch.nn.functional as F

class ReconstructionModule(nn.Module):
    def __init__(self, input_dim, output_dim, grid_size, smooth_edges=True, confidence_threshold=0.5, max_iterations=10):
        super(ReconstructionModule, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.grid_size = grid_size
        self.smooth_edges = smooth_edges
        self.confidence_threshold = confidence_threshold
        self.max_iterations = max_iterations
        self.fc = nn.Linear(input_dim, output_dim * grid_size * grid_size)

    def forward(self, features, position_logits, relation_logits=None):
        position_preds, confidence = self._get_position_predictions(position_logits)
        patches = self._rearrange_patches(features, position_preds)
        if relation_logits is not None:
            patches = self._optimize_arrangement(patches, position_preds, relation_logits, confidence)
        if self.smooth_edges:
            patches = self._smooth_edges(patches)
        reconstructed_image = self._reconstruct_image(patches)
        return reconstructed_image, confidence

    def _get_position_predictions(self, position_logits):
        # 确保我们可以安全地处理该输入
        if torch.is_floating_point(position_logits):
            # 浮点型输入，应用正常的softmax处理
            position_preds = torch.argmax(position_logits, dim=1)
            confidence = torch.max(F.softmax(position_logits, dim=1), dim=1)[0]
        else:
            # 非浮点型输入（可能是索引）
            position_preds = position_logits
            
            # 创建合适的置信度值
            if position_preds.dim() == 2:
                # 对于2D张量 [batch, num_patches]
                confidence = torch.ones(position_preds.size(0), device=position_preds.device)
            else:
                # 对于其他形状，创建匹配的置信度张量
                confidence = torch.ones(position_preds.size(0), device=position_preds.device)
        
        print(f"position_logits.shape: {position_logits.shape}, dtype: {position_logits.dtype}")
        
        return position_preds, confidence

    def _rearrange_patches(self, patches, position_preds):
        batch_size, num_patches, feature_dim = patches.size()
        
        # 将位置预测转换为长整型并限制在有效范围内
        position_preds = torch.clamp(position_preds.long(), 0, num_patches - 1)
        
        # 初始化输出张量
        rearranged_patches = torch.zeros_like(patches)
        
        # 创建索引张量用于散布操作
        batch_indices = torch.arange(batch_size, device=patches.device).view(-1, 1).repeat(1, num_patches)
        
        # 扩展位置索引以匹配特征维度
        position_indices = position_preds.unsqueeze(-1).expand(-1, -1, feature_dim)
        
        # 使用scatter操作一次性重新排列所有图像块
        rearranged_patches.scatter_(1, position_indices, patches)
        
        return rearranged_patches

    def _smooth_edges(self, patches):
        # Implement edge smoothing algorithm
        smoothed_patches = patches.clone()
        # Example smoothing operation (can be replaced with a more sophisticated algorithm)
        for i in range(1, patches.size(1) - 1):
            smoothed_patches[:, i] = (patches[:, i - 1] + patches[:, i] + patches[:, i + 1]) / 3
        return smoothed_patches

    def _optimize_arrangement(self, patches, position_preds, relation_preds, confidence):
        # Implement iterative optimization algorithm
        optimized_patches = patches.clone()
        for _ in range(self.max_iterations):
            low_confidence_indices = (confidence < self.confidence_threshold).nonzero(as_tuple=True)
            for idx in low_confidence_indices:
                # Example optimization step (can be replaced with a more sophisticated algorithm)
                optimized_patches[idx] = self._rearrange_patches(patches[idx], position_preds[idx])
        return optimized_patches

    def _reconstruct_image(self, patches):
        batch_size, num_patches, patch_dim = patches.size()
        
        # 检查是否包含类别标记（如果序列长度-1是完全平方数）
        seq_len_without_cls = num_patches - 1
        grid_size_candidate = int(seq_len_without_cls ** 0.5)
        
        if grid_size_candidate ** 2 == seq_len_without_cls:
            # 有类别标记，需要移除
            print(f"Detected class token, removing it. Original shape: {patches.shape}")
            patches = patches[:, 1:, :]  # 移除第一个标记（通常是类别标记）
            num_patches = patches.size(1)
        
        # 计算网格尺寸
        grid_size = int(num_patches ** 0.5)
        
        # 确保是完全平方数
        if grid_size ** 2 != num_patches:
            raise ValueError(f"Cannot reshape {num_patches} patches into a square grid. Expected a perfect square.")
        
        # 重建图像
        print(f"Reshaping to: [{batch_size}, {grid_size}, {grid_size}, {patch_dim}], num_patches={num_patches}")
        reconstructed_image = patches.view(batch_size, grid_size, grid_size, patch_dim)
        reconstructed_image = reconstructed_image.permute(0, 3, 1, 2).contiguous()
        return reconstructed_image
