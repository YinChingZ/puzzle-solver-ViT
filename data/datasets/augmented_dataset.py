import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class AugmentedDataset(Dataset):
    def __init__(self, data_dir, transform=None, augmentations=None):
        self.data_dir = data_dir
        self.transform = transform
        self.augmentations = augmentations
        self.image_paths = self._load_image_paths()

    def _load_image_paths(self):
        image_paths = []
        for root, _, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith(('.png', '.jpg', '.jpeg')):
                    image_paths.append(os.path.join(root, file))
        return image_paths

    def _load_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        return image

    def _preprocess_image(self, image):
        if self.transform:
            image = self.transform(image)
        return image

    def _apply_augmentations(self, image):
        if self.augmentations:
            for augmentation in self.augmentations:
                image = augmentation(image)
        return image

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = self._load_image(image_path)
        image = self._preprocess_image(image)
        image = self._apply_augmentations(image)
        return image
