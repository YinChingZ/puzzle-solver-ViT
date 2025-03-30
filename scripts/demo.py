import argparse
import torch
import matplotlib.pyplot as plt
from PIL import Image
import os
import sys
import torchvision.transforms as transforms

from models.puzzle_solver import PuzzleSolver
from utils.config import ConfigManager
from evaluators.visualizers.attention_visualizer import AttentionVisualizer
from evaluators.visualizers.reconstruction_visualizer import ReconstructionVisualizer
from data.processors.patch_generator import PatchGenerator
from data.processors.shuffler import Shuffler

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='演示拼图解算器')
    # 添加所需参数
    parser.add_argument('--image_path', required=True, help='输入图像路径')
    parser.add_argument('--checkpoint_path', help='模型检查点路径')
    parser.add_argument('--model_config', default='base', help='模型配置名称')
    parser.add_argument('--grid_size', type=int, default=4, help='拼图网格大小')
    parser.add_argument('--difficulty', default='medium', choices=['easy', 'medium', 'hard'], help='拼图难度')
    parser.add_argument('--output_path', help='结果保存路径')
    parser.add_argument('--show_attention', action='store_true', help='显示注意力可视化')
    return parser.parse_args()

def load_image(image_path, img_size=224):
    """加载并预处理输入图像"""
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image)
    return image, image_tensor

def generate_puzzle(image_tensor, grid_size=4):
    """生成拼图块"""
    patch_generator = PatchGenerator(grid_size)
    patches = patch_generator.generate_patches(image_tensor)
    return patches

def shuffle_puzzle(patches, difficulty='medium'):
    """打乱拼图块"""
    shuffler = Shuffler(difficulty)
    shuffled_patches = shuffler.shuffle_patches(patches)
    return shuffled_patches

def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    
    # 加载配置和模型
    config_manager = ConfigManager()
    config = config_manager.get_experiment_config(model_name=args.model_config)
    model = PuzzleSolver(**config)
    model.load_state_dict(torch.load(args.checkpoint_path))
    model.eval()
    
    # 加载和处理图像
    original_image, image_tensor = load_image(args.image_path, img_size=config['img_size'])
    
    # 生成和打乱拼图
    patches = generate_puzzle(image_tensor, grid_size=args.grid_size)
    shuffled_patches = shuffle_puzzle(patches, difficulty=args.difficulty)
    
    # 模型推理
    with torch.no_grad():
        position_logits, relation_logits, reconstructed_image = model(shuffled_patches)
    
    # 可视化结果
    visualizer = ReconstructionVisualizer()
    visualizer.visualize_reconstruction(original_image, shuffled_patches, reconstructed_image, grid_size=args.grid_size, save_path=args.output_path)
    
    if args.show_attention:
        attention_visualizer = AttentionVisualizer(model)
        attention_visualizer.visualize_attention(shuffled_patches, save_path=args.output_path)

if __name__ == "__main__":
    main()
