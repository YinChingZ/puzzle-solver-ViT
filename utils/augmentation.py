import torchvision.transforms as transforms
import torch
import random
import numpy as np

class PuzzleAugmentation:
    def __init__(self, img_size=224, strong_aug=True):
        self.img_size = img_size
        self.strong_aug = strong_aug
        
        # 基础变换
        self.base_transforms = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 强增强变换
        self.strong_transforms = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomApply([transforms.ColorJitter(0.3, 0.3, 0.3, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def __call__(self, image):
        if self.strong_aug and random.random() > 0.5:
            return self.strong_transforms(image)
        return self.base_transforms(image)