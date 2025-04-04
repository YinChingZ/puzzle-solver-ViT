import torch
import numpy as np

def calculate_adjacency_accuracy(pred_positions, true_positions, grid_size):
    """计算正确相邻关系的准确率"""
    batch_size = pred_positions.size(0)
    patch_count = pred_positions.size(1)
    
    total_adjacencies = 0
    correct_adjacencies = 0
    
    for b in range(batch_size):
        # 获取预测和真实位置
        p_pos = pred_positions[b].cpu().numpy()
        t_pos = true_positions[b].cpu().numpy()
        
        # 计算预测和真实的行列坐标
        p_rows, p_cols = p_pos // grid_size, p_pos % grid_size
        t_rows, t_cols = t_pos // grid_size, t_pos % grid_size
        
        # 检查每对拼图块
        for i in range(patch_count):
            for j in range(i+1, patch_count):
                # 检查真实块是否相邻
                t_adjacent = (abs(t_rows[i] - t_rows[j]) + abs(t_cols[i] - t_cols[j])) == 1
                
                if t_adjacent:
                    total_adjacencies += 1
                    
                    # 检查预测块是否相邻
                    p_adjacent = (abs(p_rows[i] - p_rows[j]) + abs(p_cols[i] - p_cols[j])) == 1
                    
                    if p_adjacent:
                        correct_adjacencies += 1
    
    # 计算准确率
    adjacency_accuracy = correct_adjacencies / total_adjacencies if total_adjacencies > 0 else 0
    
    return adjacency_accuracy