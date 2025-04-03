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

def evaluate(model, dataloader, device, output_dir=None, visualize=False, num_vis_samples=5):
    """评估模型性能"""
    model.eval()
    
    # 初始化指标变量
    metrics = {
        'position_accuracy': [],
        'relation_accuracy': [],
        'top5_accuracy': [],
        'psnr': [],
        'ssim': []
    }
    
    # 可视化计数
    vis_count = 0
    visualizer = ReconstructionVisualizer() if visualize else None
    
    with torch.no_grad():
        for batch_idx, (shuffled_patches, positions, relations, original_images) in enumerate(tqdm(dataloader)):
            # 移动数据到设备
            shuffled_patches = shuffled_patches.to(device)
            positions = positions.to(device)
            relations = relations.to(device)
            original_images = original_images.to(device) if original_images is not None else None
            
            # 模型推理
            position_logits, relation_logits, reconstructed_images = model(shuffled_patches)
            
            # 计算指标
            pos_acc = calculate_accuracy(position_logits, positions)
            rel_acc = calculate_accuracy(relation_logits, relations) if relations is not None else 0
            top5_acc = calculate_top_k_accuracy(position_logits, positions, k=5)
            
            # 图像质量评估
            psnr_val = calculate_psnr(reconstructed_images, original_images) if original_images is not None else 0
            ssim_val = calculate_ssim(reconstructed_images, original_images) if original_images is not None else 0
            
            # 记录指标
            metrics['position_accuracy'].append(pos_acc)
            metrics['relation_accuracy'].append(rel_acc)
            metrics['top5_accuracy'].append(top5_acc)
            metrics['psnr'].append(psnr_val)
            metrics['ssim'].append(ssim_val)
            
            # 可视化
            if visualize and vis_count < num_vis_samples:
                for i in range(min(shuffled_patches.size(0), num_vis_samples - vis_count)):
                    if output_dir:
                        vis_path = os.path.join(output_dir, f'sample_{vis_count}.png')
                    else:
                        vis_path = None
                        
                    visualizer.visualize_reconstruction(
                        original_images[i:i+1] if original_images is not None else None,
                        shuffled_patches[i:i+1],
                        reconstructed_images[i:i+1],
                        grid_size=int(np.sqrt(shuffled_patches.size(1))),
                        save_path=vis_path
                    )
                    vis_count += 1
                    if vis_count >= num_vis_samples:
                        break
    
    # 计算平均指标
    results = {
        'position_accuracy': np.mean(metrics['position_accuracy']).item(),
        'relation_accuracy': np.mean(metrics['relation_accuracy']).item(),
        'top5_accuracy': np.mean(metrics['top5_accuracy']).item(),
        'psnr': np.mean(metrics['psnr']).item(),
        'ssim': np.mean(metrics['ssim']).item()
    }
    
    # 保存结果
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'evaluation_results.json'), 'w') as f:
            json.dump(results, f, indent=4)
            
        # 创建简单的结果可视化
        plt.figure(figsize=(10, 6))
        plt.bar(results.keys(), results.values())
        plt.title('Evaluation Metrics')
        plt.savefig(os.path.join(output_dir, 'metrics_summary.png'))
    
    return results

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
    
    # 加载配置
    config_manager = ConfigManager()
    model_config = config_manager.load_config("model", args.model_config)
    data_config = config_manager.load_config("data", args.data_config)
    
    # 加载模型
    model = PuzzleSolver(**model_config)
    model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
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

if __name__ == '__main__':
    main()
