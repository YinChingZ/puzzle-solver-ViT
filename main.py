import argparse
import json
import os
import torch
from models.puzzle_solver import PuzzleSolver
from data.datasets.multi_domain_dataset import MultiDomainDataset
from trainers.curriculum_trainer import CurriculumTrainer
from utils.logging_utils import setup_logger, log_training_progress
from utils.optimization import get_optimizer, get_scheduler

def main(config):
    # Set up logging
    logger = setup_logger('train_logger', os.path.join(config['log_dir'], 'train.log'))

    # Create directories for logs and checkpoints
    os.makedirs(config['log_dir'], exist_ok=True)
    os.makedirs(config['checkpoint_dir'], exist_ok=True)

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
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

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

    # Initialize trainer
    trainer = CurriculumTrainer(model, optimizer, criterion, train_loader, val_loader, config['num_epochs'], config['device'], config['difficulty_scheduler'])

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
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file.')
    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)

    main(config)
