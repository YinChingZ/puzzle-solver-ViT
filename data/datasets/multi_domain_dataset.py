import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class MultiDomainDataset(Dataset):
    def __init__(self, data_dirs, transform=None, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
        self.data_dirs = data_dirs
        self.transform = transform
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
        image = self._load_image(image_path)
        image = self._preprocess_image(image)
        return image
