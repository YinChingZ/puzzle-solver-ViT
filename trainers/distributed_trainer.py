import torch
import os
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from trainers.base_trainer import BaseTrainer

class DistributedTrainer(BaseTrainer):
    def __init__(self, model, optimizer, criterion, train_dataset, val_dataset, batch_size=32, num_epochs=100, device='cuda', world_size=1, rank=0):
        super(DistributedTrainer, self).__init__(model, optimizer, criterion, train_dataset, val_dataset, batch_size, num_epochs, device)
        self.world_size = world_size
        self.rank = rank
        self.model = DDP(self.model, device_ids=[self.rank])

    def train(self):
        for epoch in range(self.num_epochs):
            self.model.train()
            train_loss = 0.0
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()

            val_loss = self.validate()

            if self.rank == 0:
                print(f'Epoch [{epoch+1}/{self.num_epochs}], Train Loss: {train_loss/len(self.train_loader)}, Val Loss: {val_loss/len(self.val_loader)}')

    def validate(self):
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                val_loss += loss.item()

        return val_loss

    def save_checkpoint(self, checkpoint_path):
        if self.rank == 0:
            state = {
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'epoch': self.num_epochs
            }
            torch.save(state, checkpoint_path)

    def load_checkpoint(self, checkpoint_path):
        if os.path.isfile(checkpoint_path):
            state = torch.load(checkpoint_path)
            self.model.load_state_dict(state['model'])
            self.optimizer.load_state_dict(state['optimizer'])
            self.num_epochs = state['epoch']
        else:
            print(f'No checkpoint found at {checkpoint_path}')
