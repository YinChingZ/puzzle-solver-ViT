import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import torch
import os
import json
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from models.puzzle_solver import PuzzleSolver
from data.datasets.multi_domain_dataset import MultiDomainDataset
from utils.config import ConfigManager
from utils.metrics import calculate_accuracy, calculate_top_k_accuracy, calculate_psnr, calculate_ssim
from evaluators.visualizers.reconstruction_visualizer import ReconstructionVisualizer

# 更新评估函数

def evaluate(model, dataloader, device, output_dir=None, visualize=False, num_vis_samples=5):
    """评估模型性能"""
    model.eval()
    
    # 初始化指标变量
    metrics = {
        'position_accuracy': [],
        'relation_accuracy': [],
        'top5_accuracy': [],
        'psnr': [],
        'ssim': [],
        'adjacency_accuracy': [],  # 新增邻接准确率
        'grid_size_metrics': {}    # 按网格大小分组的指标
    }
    
    # 可视化计数
    vis_count = 0
    visualizer = ReconstructionVisualizer() if visualize else None
    
    with torch.no_grad():
        for images, positions in tqdm(dataloader, desc="评估中"):
            # 获取当前批次的网格大小
            patch_count = positions.size(1)
            grid_size = int(np.sqrt(patch_count))
            
            # 如果网格大小不在记录中，添加它
            if grid_size not in metrics['grid_size_metrics']:
                metrics['grid_size_metrics'][grid_size] = {
                    'position_accuracy': [],
                    'relation_accuracy': [],
                    'adjacency_accuracy': []
                }
            
            images = images.to(device)
            positions = positions.to(device)
            
            # 模型推理
            position_logits, relation_logits, reconstructed = model(images)
            
            # 计算位置预测准确率
            position_preds = torch.argmax(position_logits, dim=1)
            accuracy = calculate_accuracy(position_preds, positions)
            metrics['position_accuracy'].append(accuracy)
            metrics['grid_size_metrics'][grid_size]['position_accuracy'].append(accuracy)
            
            # 计算Top-5准确率
            top5_acc = calculate_top_k_accuracy(position_logits, positions, k=5)
            metrics['top5_accuracy'].append(top5_acc)
            
            # 计算邻接准确率
            adj_acc = calculate_adjacency_accuracy(position_preds, positions, grid_size)
            metrics['adjacency_accuracy'].append(adj_acc)
            metrics['grid_size_metrics'][grid_size]['adjacency_accuracy'].append(adj_acc)
            
            # ... 原有代码 ...
    
    # 计算平均指标
    result = {}
    for key in ['position_accuracy', 'relation_accuracy', 'top5_accuracy', 'adjacency_accuracy', 'psnr', 'ssim']:
        if metrics[key]:
            result[key] = sum(metrics[key]) / len(metrics[key]) if len(metrics[key]) > 0 else 0
    
    # 处理按网格大小的指标
    result['grid_size_metrics'] = {}
    for grid_size, grid_metrics in metrics['grid_size_metrics'].items():
        result['grid_size_metrics'][grid_size] = {}
        for key, values in grid_metrics.items():
            result['grid_size_metrics'][grid_size][key] = sum(values) / len(values) if len(values) > 0 else 0
    
    return result

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Evaluate the puzzle solver model')
    parser.add_argument('--model_config', default='base', help='Model configuration name')
    parser.add_argument('--data_config', default='default', help='Data configuration name')
    parser.add_argument('--checkpoint_path', required=True, help='Path to model checkpoint')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for evaluation')
    parser.add_argument('--output_dir', default='evaluation_results', help='Directory to save results')
    parser.add_argument('--visualize', action='store_true', help='Generate visualizations')
    parser.add_argument('--num_vis_samples', type=int, default=5, help='Number of samples to visualize')
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 检查检查点
    # inspect_checkpoint(args.checkpoint_path, device)
    
    # 加载配置
    config_manager = ConfigManager()
    model_config = config_manager.load_config("model", args.model_config)
    data_config = config_manager.load_config("data", args.data_config)
    
    # 加载模型
    model = PuzzleSolver(**model_config)
    # 修改这一行，从检查点中提取 'model' 键
    checkpoint = torch.load(args.checkpoint_path, map_location=device, weights_only=True)
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        print("检测到标准检查点格式，正在提取模型状态...")
        model.load_state_dict(checkpoint['model'])
    else:
        # 尝试直接加载（以防某些检查点采用不同格式）
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    print(f"Model loaded from {args.checkpoint_path}")
    
    # 准备数据集
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((model_config.get('img_size', 224), model_config.get('img_size', 224))),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = MultiDomainDataset(
        data_dirs=data_config.get('test_dirs', []),
        transform=transform
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    print(f"Test dataset loaded with {len(test_dataset)} samples")
    
    # 评估模型
    print("Starting evaluation...")
    results = evaluate(
        model, 
        test_loader, 
        device, 
        output_dir=args.output_dir,
        visualize=args.visualize, 
        num_vis_samples=args.num_vis_samples
    )
    
    # 打印结果
    print("\nEvaluation Results:")
    print(f"Position Prediction Accuracy: {results['position_accuracy']:.4f}")
    print(f"Relation Prediction Accuracy: {results['relation_accuracy']:.4f}")
    print(f"Top-5 Accuracy: {results['top5_accuracy']:.4f}")
    print(f"PSNR: {results['psnr']:.2f} dB")
    print(f"SSIM: {results['ssim']:.4f}")
    print(f"\nDetailed results saved to {args.output_dir}")
    
    return results

def inspect_checkpoint(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if 'config' in checkpoint:
        print("配置信息:", checkpoint['config'])
    print("模型状态字典键:", checkpoint['model'].keys())
    # 打印几个关键层的形状
    for key in ['transformer_encoder.blocks.0.mlp.fc1.weight', 
                'position_head.fc.weight']:
        if key in checkpoint['model']:
            print(f"{key} 形状: {checkpoint['model'][key].shape}")

if __name__ == '__main__':
    main()
