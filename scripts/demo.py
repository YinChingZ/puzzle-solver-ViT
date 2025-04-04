import argparse
import torch
import matplotlib.pyplot as plt
from PIL import Image
import os
import sys
import torchvision.transforms as transforms

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
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

        # 过滤掉不支持的参数
        if 'cnn_feature_extractor' in model_config:
            del model_config['cnn_feature_extractor']
        # 可能还需要过滤其他不支持的参数
        for key in list(model_config.keys()):
            if key not in ["img_size", "patch_size", "num_classes", "embed_dim", 
                        "depth", "num_heads", "mlp_ratio", "qkv_bias", "qk_scale", 
                        "drop_rate", "attn_drop_rate", "drop_path_rate", "norm_layer", 
                        "grid_size"]:
                del model_config[key]

        # 初始化模型
        config_manager = ConfigManager()
        model_config = config_manager.load_config("model", "custom")  # 使用与训练相同的配置

        # 初始化模型
        model = PuzzleSolver(**model_config)

        # 修改加载模型权重的部分（大约在第92行）
        if args.checkpoint_path:
            if not os.path.exists(args.checkpoint_path):
                raise FileNotFoundError(f"模型检查点不存在: {args.checkpoint_path}")
            
            # 加载检查点
            checkpoint = torch.load(args.checkpoint_path, map_location=device, weights_only=True)
            
            # 检查检查点格式并适当提取模型权重
            if isinstance(checkpoint, dict) and 'model' in checkpoint:
                print("检测到标准检查点格式，正在提取模型状态...")
                model.load_state_dict(checkpoint['model'])
            else:
                # 尝试直接加载
                model.load_state_dict(checkpoint)
            
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
        
        print(f"Patches shape: {patches.shape}")
        print(f"Positions shape: {positions.shape}")
        print(f"Shuffled patches shape: {shuffled_patches.shape}")
        print(f"Shuffled positions shape: {shuffled_positions.shape}")

        # 模型推理
        # 修改模型推理部分（大约在第137行）
        with torch.no_grad():
            # 重塑拼图块张量以适应卷积层输入
            b, n, c, h, w = shuffled_patches.shape
            shuffled_patches_reshaped = shuffled_patches.view(b*n, c, h, w).to(device)
            
            # 修改位置标签以匹配批处理后的大小，并确保在同一设备上
            positions_reshaped = positions.flatten().to(device)
            
            # 传递重塑后的数据到模型
            position_logits, relation_logits, reconstructed_image = model(shuffled_patches_reshaped)
            
            # 打印形状信息以帮助调试
            print(f"Position logits shape: {position_logits.shape}")
            print(f"Positions_reshaped shape: {positions_reshaped.shape}")
            
            # 计算位置预测准确率
            position_preds = torch.argmax(position_logits, dim=1)
            accuracy = (position_preds == positions_reshaped).float().mean().item()
            print(f"位置预测准确率: {accuracy:.2%}")
        
        # 修改可视化结果部分
        print("生成可视化结果...")
        visualizer = ReconstructionVisualizer()

        # 将拼图块组合成完整图像
        def assemble_image_from_patches(patches, grid_size):
            """将拼图块组合成完整图像"""
            b, n, c, h, w = patches.shape
            rows = []
            for i in range(grid_size):
                row_patches = []
                for j in range(grid_size):
                    idx = i * grid_size + j
                    row_patches.append(patches[0, idx])
                row = torch.cat(row_patches, dim=2)  # 水平连接
                rows.append(row)
            assembled = torch.cat(rows, dim=1)  # 垂直连接
            return assembled.unsqueeze(0)  # 添加批次维度 [1, C, H, W]

        # 组装原始图像、打乱的图像和重建的图像
        original_assembled = assemble_image_from_patches(patches, args.grid_size)
        shuffled_assembled = assemble_image_from_patches(shuffled_patches, args.grid_size)

        # 确保所有输入都在同一设备上
        image_tensor_cpu = image_tensor.cpu()
        original_assembled_cpu = original_assembled.cpu()
        shuffled_assembled_cpu = shuffled_assembled.cpu()
        reconstructed_image_cpu = reconstructed_image.cpu()

        # 传递组合后的图像给可视化器
        visualizer.visualize_reconstruction(
            image_tensor_cpu,  # 原始图像
            shuffled_assembled_cpu,  # 打乱的拼图
            reconstructed_image_cpu,  # 重建的图像
            grid_size=args.grid_size,
            save_path=args.output_path
        )
        
        print(f"Reconstructed image shape: {reconstructed_image.shape}")
        
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
