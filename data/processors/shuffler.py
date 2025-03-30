import torch
import numpy as np
import random

class Shuffler:
    def __init__(self, difficulty='medium', seed=None):
        """
        初始化拼图块打乱器
        
        参数:
            difficulty: 难度级别 ('easy', 'medium', 'hard')
            seed: 随机数种子，用于可重复的结果
        """
        self.difficulty = difficulty
        self.seed = seed
        
        # 设置不同难度级别对应的打乱程度
        self.difficulty_mapping = {
            'easy': 0.3,    # 容易：30%的块被打乱
            'medium': 0.6,  # 中等：60%的块被打乱
            'hard': 1.0     # 困难：100%的块被打乱
        }
    
    def shuffle_patches(self, patches, positions=None, return_positions=False):
        """
        打乱拼图块
        
        参数:
            patches: 拼图块张量 [B, N, C, H, W]
            positions: 块的原始位置索引 [B, N]
            return_positions: 是否返回打乱后的位置
            
        返回:
            shuffled_patches: 打乱后的拼图块
            shuffled_positions: (如果return_positions=True) 打乱后的位置索引
        """
        # 设置随机种子（如果提供）
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
        
        # 获取批次大小和块数量
        batch_size, num_patches = patches.shape[0], patches.shape[1]
        
        # 如果没有提供位置信息，创建序列位置
        if positions is None:
            positions = torch.arange(num_patches).unsqueeze(0).repeat(batch_size, 1)
        
        # 根据难度确定要打乱的块比例
        swap_ratio = self.difficulty_mapping.get(self.difficulty, 0.6)  # 默认中等难度
        num_swaps = int(num_patches * swap_ratio)
        
        # 为每个批次创建打乱的块
        shuffled_patches = []
        shuffled_positions = []
        
        for b in range(batch_size):
            # 复制原始块和位置
            batch_patches = patches[b].clone()
            batch_positions = positions[b].clone()
            
            # 生成随机交换索引
            indices = list(range(num_patches))
            random.shuffle(indices)
            indices = indices[:num_swaps]  # 根据难度选择要打乱的块数量
            
            # 生成目标索引
            targets = indices.copy()
            random.shuffle(targets)
            
            # 执行交换
            for i, j in zip(indices, targets):
                # 交换块
                temp_patch = batch_patches[i].clone()
                batch_patches[i] = batch_patches[j]
                batch_patches[j] = temp_patch
                
                # 交换位置信息
                temp_pos = batch_positions[i].clone()
                batch_positions[i] = batch_positions[j]
                batch_positions[j] = temp_pos
            
            shuffled_patches.append(batch_patches)
            shuffled_positions.append(batch_positions)
        
        # 组合批次
        shuffled_patches = torch.stack(shuffled_patches)
        shuffled_positions = torch.stack(shuffled_positions)
        
        if return_positions:
            return shuffled_patches, shuffled_positions
        return shuffled_patches
