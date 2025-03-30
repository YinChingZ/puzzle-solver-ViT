# scripts/train.py
"""训练脚本，支持通过独立配置参数进行灵活设置，适用于实验和开发"""

import argparse
import os
import torch
from torch.utils.data import DataLoader
from models.puzzle_solver import PuzzleSolver
from data.datasets.multi_domain_dataset import MultiDomainDataset
from trainers.curriculum_trainer import CurriculumTrainer
from utils.logging_utils import setup_logger, log_training_progress
from utils.optimization import get_optimizer, get_scheduler
from utils.config import ConfigManager
from torchvision import transforms

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
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

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
