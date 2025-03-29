import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionHead(nn.Module):
    def __init__(self, input_dim, num_positions):
        super(PositionHead, self).__init__()
        self.fc = nn.Linear(input_dim, num_positions)

    def forward(self, x):
        x = self.fc(x)
        return x
