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
    
    def shuffle_patches(self, patches, positions=None, return_positions=True):
        """
        打乱拼图块
        
        参数:
            patches: 拼图块张量
            positions: 块的原始位置索引
            return_positions: 是否返回打乱后的位置
            
        返回:
            shuffled_patches: 打乱后的拼图块
            shuffled_positions: (如果return_positions=True) 打乱后的位置索引
        """
        # 1. 设置随机种子（如果提供）
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
        
        # 2. 获取块数量和批次大小
        num_patches = patches.size(0)
        batch_size = patches.size(1) if patches.dim() == 4 else 1
        
        # 3. 根据难度确定要打乱的块比例
        shuffle_ratio = self.difficulty_mapping.get(self.difficulty, 0.6)
        num_shuffle = int(num_patches * shuffle_ratio)
        
        # 4. 生成随机交换索引
        shuffle_indices = np.random.permutation(num_patches)[:num_shuffle]
        
        # 5. 执行打乱
        shuffled_patches = patches.clone()
        shuffled_positions = positions.clone() if positions is not None else None
        
        for idx in shuffle_indices:
            swap_idx = np.random.choice(num_patches)
            shuffled_patches[idx], shuffled_patches[swap_idx] = shuffled_patches[swap_idx], shuffled_patches[idx]
            if shuffled_positions is not None:
                shuffled_positions[idx], shuffled_positions[swap_idx] = shuffled_positions[swap_idx], shuffled_positions[idx]
        
        # 6. 返回结果
        if return_positions:
            return shuffled_patches, shuffled_positions
        return shuffled_patches
