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
    parser.add_argument('--attention_head', type=int, default=0, help='要可视化的注意力头')
    return parser.parse_args()

def load_image(image_path, img_size=224):
    """加载并预处理输入图像"""
    # 检查文件是否存在
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图像文件不存在: {image_path}")
        
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)  # 添加批次维度 [1, C, H, W]
    return image, image_tensor

def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    try:
        # 加载配置和模型
        config_manager = ConfigManager()
        model_config = config_manager.load_config("model", args.model_config)
        
        # 合并grid_size到配置
        model_config['grid_size'] = args.grid_size
        
        # 初始化模型
        model = PuzzleSolver(**model_config)
        
        # 如果提供了检查点，加载权重
        if args.checkpoint_path:
            if not os.path.exists(args.checkpoint_path):
                raise FileNotFoundError(f"模型检查点不存在: {args.checkpoint_path}")
            
            model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
            print(f"成功从 {args.checkpoint_path} 加载模型权重")
        else:
            print("警告: 未提供模型检查点，使用随机初始化的权重")
            
        model = model.to(device)
        model.eval()
        
        # 加载和处理图像
        original_image, image_tensor = load_image(args.image_path, img_size=model_config.get('img_size', 224))
        image_tensor = image_tensor.to(device)
        
        print(f"成功加载图像: {args.image_path}")
        
        # 生成和打乱拼图
        patch_generator = PatchGenerator(grid_size=args.grid_size)
        patches, positions = patch_generator.generate_patches(image_tensor, return_positions=True)
        
        shuffler = Shuffler(difficulty=args.difficulty)
        shuffled_patches, shuffled_positions = shuffler.shuffle_patches(patches, positions, return_positions=True)
        
        print(f"已生成 {args.grid_size}x{args.grid_size} 网格的拼图，难度: {args.difficulty}")
        
        # 模型推理
        with torch.no_grad():
            position_logits, relation_logits, reconstructed_image = model(shuffled_patches)
            
            # 计算位置预测准确率
            position_preds = torch.argmax(position_logits, dim=1)
            accuracy = (position_preds == positions).float().mean().item()
            print(f"位置预测准确率: {accuracy:.2%}")
        
        # 可视化结果
        print("生成可视化结果...")
        visualizer = ReconstructionVisualizer()
        visualizer.visualize_reconstruction(
            image_tensor,
            shuffled_patches,
            reconstructed_image,
            grid_size=args.grid_size,
            save_path=args.output_path
        )
        
        # 如果需要，显示注意力图
        if args.show_attention:
            print("生成注意力可视化...")
            attention_visualizer = AttentionVisualizer(model)
            attention_path = None
            if args.output_path:
                attention_path = args.output_path.replace('.png', '_attention.png')
            attention_visualizer.visualize_attention(
                shuffled_patches,
                head=args.attention_head,
                save_path=attention_path
            )
            
        print("演示完成!")
        
    except Exception as e:
        print(f"错误: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
