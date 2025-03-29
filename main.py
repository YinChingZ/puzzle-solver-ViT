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

    # Load datasets
    train_dataset = MultiDomainDataset(config['data']['train'])
    val_dataset = MultiDomainDataset(config['data']['val'])

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

    # Initialize model
    model = PuzzleSolver(config['model'])

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
