import os
import numpy as np
from PIL import Image
import torch 
from torch.utils.data import Dataset
from torchvision import transforms

class MultiDomainDataset(Dataset):
    def __init__(self, data_dirs, transform=None, grid_size=4, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
        self.data_dirs = data_dirs
        self.transform = transform
        self.grid_size = grid_size  # 添加这一行
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.image_paths = self._load_image_paths()
        self.train_paths, self.val_paths, self.test_paths = self._split_dataset()

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
        image_path = self.image_paths[idx]
        original_image = self._load_image(image_path)
        
        # 创建拼图
        scrambled_image, position_labels = self._create_puzzle(original_image)
        
        # 应用变换
        if self.transform:
            scrambled_image = self.transform(scrambled_image)
        
        return scrambled_image, position_labels
