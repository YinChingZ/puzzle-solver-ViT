import torch
import torch.nn as nn
import torch.nn.functional as F

class ReconstructionModule(nn.Module):
    def __init__(self, input_dim, output_dim, grid_size):
        super(ReconstructionModule, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.grid_size = grid_size
        self.fc = nn.Linear(input_dim, output_dim * grid_size * grid_size)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, self.output_dim, self.grid_size, self.grid_size)
        return x
