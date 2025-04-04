# scripts/train.py
"""训练脚本，支持通过独立配置参数进行灵活设置，适用于实验和开发"""

import argparse
import os
import sys
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.puzzle_solver import PuzzleSolver
from data.datasets.multi_domain_dataset import MultiDomainDataset
from trainers.curriculum_trainer import CurriculumTrainer
from utils.logging_utils import setup_logger, log_training_progress
from utils.optimization import get_optimizer, get_scheduler
from utils.config import ConfigManager
from torchvision import transforms

def create_dataloaders(train_dataset, val_dataset, batch_size, num_workers=4):
    """创建优化的数据加载器"""
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,  # 并行加载
        pin_memory=True,          # 加速GPU传输
        prefetch_factor=2,        # 预取倍数
        persistent_workers=True   # 保持工作进程
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True
    )
    
    return train_loader, val_loader

def main(args):
    # 加载配置
    config_manager = ConfigManager()
    config = config_manager.get_experiment_config(
        model_name=args.model_config,
        data_name=args.data_config,
        training_name=args.training_config
    )

    # Set up logging
    logger = setup_logger('train_logger', os.path.join(config['log_dir'], 'train.log'))

    # Create directories for logs and checkpoints
    os.makedirs(config['log_dir'], exist_ok=True)
    os.makedirs(config['checkpoint_dir'], exist_ok=True)

    # Define the transform variable
    transform = transforms.Compose([
        transforms.Resize((config['img_size'], config['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load datasets
    train_dataset = MultiDomainDataset(
        data_dirs=config['data']['train_dirs'],
        transform=transform
    )
    val_dataset = MultiDomainDataset(
        data_dirs=config['data']['val_dirs'],
        transform=transform
    )

    # Create data loaders
    # 根据系统CPU核心数选择合适的num_workers
    num_workers = min(8, os.cpu_count() or 4)

    train_loader, val_loader = create_dataloaders(
        train_dataset, 
        val_dataset, 
        config['batch_size'], 
        num_workers=num_workers
    )

    # Initialize model
    model = PuzzleSolver(
        img_size=config["img_size"],
        patch_size=config["patch_size"],
        num_classes=config["num_classes"],
        embed_dim=config["embed_dim"],
        depth=config["depth"],
        num_heads=config["num_heads"],
        # 其他参数...
    )

    # Set up optimizer and scheduler
    optimizer = get_optimizer(model, config['optimizer']['name'], config['optimizer']['lr'], config['optimizer']['weight_decay'])
    scheduler = get_scheduler(optimizer, config['scheduler']['name'], config['scheduler']['T_max'], config['scheduler']['eta_min'])

    # Set up criterion
    criterion = torch.nn.CrossEntropyLoss()

    # Handle curriculum learning parameters
    difficulty_scheduler = None
    if config.get('curriculum', {}).get('enabled', False):
        difficulty_scheduler = config.get('curriculum', {}).get('stages', [])
        print(difficulty_scheduler)

    # Initialize trainer
    trainer = CurriculumTrainer(model, optimizer, criterion, train_loader, val_loader, config['num_epochs'], config['device'], difficulty_scheduler)

    # Train the model
    for epoch in range(config['num_epochs']):
        trainer.train()
        val_loss = trainer.validate()
        log_training_progress(logger, epoch, trainer.train_loss, val_loss)
        scheduler.step()

        # Save checkpoint
        if (epoch + 1) % config['checkpoint_interval'] == 0:
            trainer.save_checkpoint(os.path.join(config['checkpoint_dir'], f'checkpoint_epoch_{epoch+1}.pth'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the Vision Transformer-based puzzle solver.')
    parser.add_argument('--model_config', type=str, default='base', help='Name of the model configuration.')
    parser.add_argument('--data_config', type=str, default='default', help='Name of the data configuration.')
    parser.add_argument('--training_config', type=str, default='default', help='Name of the training configuration.')
    args = parser.parse_args()

    main(args)
