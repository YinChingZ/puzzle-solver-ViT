import os
import numpy as np
from PIL import Image
import torch 
from torch.utils.data import Dataset
from torchvision import transforms
from data.processors.patch_generator import PatchGenerator

class MultiDomainDataset(Dataset):
    def __init__(self, data_dirs, transform=None, img_size=224):
        self.data_dirs = data_dirs if isinstance(data_dirs, list) else [data_dirs]
        self.transform = transform
        self.img_size = img_size
        self.image_paths = self._load_image_paths()
        
        # 添加拼图块生成器支持
        self.grid_size = 4  # 默认网格大小
        self.patch_generator = PatchGenerator(grid_size=self.grid_size)
    
    def set_grid_size(self, grid_size):
        """更新网格大小以支持课程学习"""
        self.grid_size = grid_size
        self.patch_generator = PatchGenerator(grid_size=grid_size)
        print(f"数据集网格大小已更新为: {grid_size}x{grid_size}")

    def _load_image_paths(self):
        image_paths = []
        for data_dir in self.data_dirs:
            for root, _, files in os.walk(data_dir):
                for file in files:
                    if file.endswith(('.png', '.jpg', '.jpeg')):
                        image_paths.append(os.path.join(root, file))
        return image_paths

    def _split_dataset(self):
        np.random.shuffle(self.image_paths)
        total_images = len(self.image_paths)
        train_end = int(total_images * self.train_ratio)
        val_end = train_end + int(total_images * self.val_ratio)
        train_paths = self.image_paths[:train_end]
        val_paths = self.image_paths[train_end:val_end]
        test_paths = self.image_paths[val_end:]
        return train_paths, val_paths, test_paths
    
    def _create_puzzle(self, image):
        """将图像切分为grid_size×grid_size的网格，并随机打乱"""
        width, height = image.size
        piece_width = width // self.grid_size
        piece_height = height // self.grid_size
        
        # 切分图像
        pieces = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                box = (j * piece_width, i * piece_height, 
                      (j + 1) * piece_width, (i + 1) * piece_height)
                piece = image.crop(box)
                pieces.append(piece)
        
        # 生成原始位置索引和打乱的位置索引
        original_indices = list(range(self.grid_size * self.grid_size))
        scrambled_indices = original_indices.copy()
        np.random.shuffle(scrambled_indices)
        
        # 按打乱的顺序重组图像
        scrambled_pieces = [pieces[i] for i in scrambled_indices]
        
        # 创建打乱后的完整图像
        scrambled_image = Image.new('RGB', (width, height))
        for idx, piece in enumerate(scrambled_pieces):
            i = idx // self.grid_size
            j = idx % self.grid_size
            box = (j * piece_width, i * piece_height, 
                  (j + 1) * piece_width, (i + 1) * piece_height)
            scrambled_image.paste(piece, box)
        
        # 计算每个打乱后的位置应该在的正确位置
        position_labels = torch.tensor([scrambled_indices.index(i) for i in original_indices])
        
        return scrambled_image, position_labels

    def _load_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        return image

    def _preprocess_image(self, image):
        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 获取图像路径
        image_path = self.image_paths[idx]
        
        try:
            # 尝试加载图像
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"无法加载图像 {image_path}: {str(e)}")
            # 创建一个默认的黑色图像作为替代
            image = Image.new('RGB', (self.img_size, self.img_size), color='black')
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
        
        # 生成拼图块并立即重组为完整图像
        if hasattr(self, 'patch_generator'):
            patches, positions = self.patch_generator.generate_patches(image)
            
            # 重组拼图块为完整图像
            batch_size, num_patches, channels, patch_h, patch_w = patches.shape
            grid_size = int(num_patches ** 0.5)
            img_h, img_w = grid_size * patch_h, grid_size * patch_w
            
            # 创建重组图像
            recomposed_image = torch.zeros(batch_size, channels, img_h, img_w, device=patches.device)
            
            # 填充图像
            for b in range(batch_size):
                for i in range(grid_size):
                    for j in range(grid_size):
                        patch_idx = i * grid_size + j
                        y_start = i * patch_h
                        y_end = (i+1) * patch_h
                        x_start = j * patch_w
                        x_end = (j+1) * patch_w
                        
                        recomposed_image[b, :, y_start:y_end, x_start:x_end] = patches[b, patch_idx]
            
            # 返回重组的图像和位置信息
            return recomposed_image, positions
        
        # 如果没有patch_generator，返回原始图像和默认位置
        dummy_positions = torch.zeros(1)  # 创建一个默认位置
        return image, dummy_positions