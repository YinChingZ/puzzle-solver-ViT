import torch
import torch.nn as nn
import torch.nn.functional as F

class RelationHead(nn.Module):
    def __init__(self, input_dim, num_relations):
        super(RelationHead, self).__init__()
        self.fc = nn.Linear(input_dim, num_relations)

    def forward(self, x):
        x = self.fc(x)
        return x
