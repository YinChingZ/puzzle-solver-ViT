import os
import random
import shutil
from pathlib import Path

def split_dataset(source_dir, output_base_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """将源目录中的图像分割到训练、验证和测试集中"""
    random.seed(seed)
    
    # 创建输出目录
    train_dir = os.path.join(output_base_dir, "train")
    val_dir = os.path.join(output_base_dir, "val")
    test_dir = os.path.join(output_base_dir, "test")
    
    for directory in [train_dir, val_dir, test_dir]:
        os.makedirs(directory, exist_ok=True)
    
    # 收集图像文件
    extensions = ['.jpg', '.jpeg', '.png']
    image_files = []
    
    for ext in extensions:
        image_files.extend(list(Path(source_dir).glob(f"**/*{ext}")))
    
    # 随机打乱
    random.shuffle(image_files)
    
    # 计算分割点
    total_images = len(image_files)
    train_end = int(total_images * train_ratio)
    val_end = train_end + int(total_images * val_ratio)
    
    # 分割并复制文件
    for i, img_path in enumerate(image_files):
        if i < train_end:
            dest_dir = train_dir
        elif i < val_end:
            dest_dir = val_dir
        else:
            dest_dir = test_dir
            
        dest_path = os.path.join(dest_dir, img_path.name)
        shutil.copy2(img_path, dest_path)
        
    # 打印统计信息
    print(f"数据集准备完成:")
    print(f"总图像: {total_images}")
    print(f"训练集: {train_end} 图像")
    print(f"验证集: {val_end - train_end} 图像")
    print(f"测试集: {total_images - val_end} 图像")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="准备拼图解算器训练数据集")
    parser.add_argument("--source", required=True, help="源图像目录")
    parser.add_argument("--output", default="data", help="输出数据目录")
    parser.add_argument("--train", type=float, default=0.8, help="训练集比例")
    parser.add_argument("--val", type=float, default=0.1, help="验证集比例")
    parser.add_argument("--test", type=float, default=0.1, help="测试集比例")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    
    args = parser.parse_args()
    
    # 验证比例和为1
    total = args.train + args.val + args.test
    if abs(total - 1.0) > 0.001:
        print(f"错误: 训练、验证和测试比例之和应为1.0，当前为{total}")
        exit(1)
    
    split_dataset(args.source, args.output, args.train, args.val, args.test, args.seed)