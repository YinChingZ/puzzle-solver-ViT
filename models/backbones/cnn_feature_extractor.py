import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNFeatureExtractor(nn.Module):
    def __init__(self, input_channels=3, feature_dim=256, grid_size=4, use_bn=True, dropout=0.1):
        super(CNNFeatureExtractor, self).__init__()
        self.use_bn = use_bn
        self.dropout = dropout
        self.grid_size = grid_size

        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64) if use_bn else nn.Identity()
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128) if use_bn else nn.Identity()
        self.conv3 = nn.Conv2d(128, feature_dim, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(feature_dim) if use_bn else nn.Identity()

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout_layer = nn.Dropout(dropout)

        self.adjust_pooling_layers()

    def adjust_pooling_layers(self):
        self.pooling_layers = []
        for _ in range(self.grid_size):
            self.pooling_layers.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
        self.pooling_layers = nn.ModuleList(self.pooling_layers)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.dropout_layer(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout_layer(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = self.dropout_layer(x)

        x += residual  # Residual connection

        # Adjust output format to ensure compatibility with ViT encoder
        x = x.view(x.size(0), -1)

        return x
