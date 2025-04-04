import argparse
import torch
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.puzzle_solver import PuzzleSolver
from utils.config import ConfigManager
from evaluators.visualizers.attention_visualizer import AttentionVisualizer
from data.processors.patch_generator import PatchGenerator

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='可视化模型注意力')
    parser.add_argument('--model_config', type=str, default='enhanced', help='模型配置名')
    parser.add_argument('--checkpoint_path', required=True, help='检查点路径')
    parser.add_argument('--image_path', required=True, help='输入图像路径')
    parser.add_argument('--output_path', help='输出图像保存路径')
    parser.add_argument('--head', type=int, default=0, help='要可视化的注意力头')
    args = parser.parse_args()
    
    # 初始化模型
    config_manager = ConfigManager()
    model_config = config_manager.load_config("model", args.model_config)
    
    model = PuzzleSolver(**model_config)
    
    # 加载检查点
    checkpoint = torch.load(args.checkpoint_path, map_location='cuda')
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to('cuda')
    model.eval()
    
    # 加载图像
    from data.utils.image_utils import load_image
    image, image_tensor = load_image(args.image_path, img_size=model_config.get('img_size', 224))
    image_tensor = image_tensor.to('cuda')
    
    # 生成拼图块
    patch_generator = PatchGenerator(grid_size=model_config.get('grid_size', 12))
    patches, _ = patch_generator.generate_patches(image_tensor)
    
    # 可视化注意力
    visualizer = AttentionVisualizer(model)
    visualizer.visualize_attention_map(model, patches, save_path=args.output_path, head_idx=args.head)
    
    print(f"注意力可视化完成！")

if __name__ == '__main__':
    main()