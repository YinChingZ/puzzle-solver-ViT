import os
import sys
import torch
import numpy as np
import random
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.puzzle_solver import PuzzleSolver
from models.modules.feature_decoder import FeatureDecoder

# Configuration
checkpoint_path = 'checkpoints/rtx3050/checkpoint_epoch_30.pth'
test_image_path = 'data/train/unsplash_00001.jpg'
output_dir = 'test_results'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Function to create shuffled image
def create_shuffled_image(image_tensor, grid_size=12, shuffle_ratio=0.7):
    """
    Create a shuffled puzzle from an image tensor
    
    Args:
        image_tensor: Input image tensor [B,C,H,W]
        grid_size: Number of grid cells in each dimension
        shuffle_ratio: Portion of pieces to shuffle (0-1)
        
    Returns:
        shuffled_image: Shuffled image tensor [B,C,H,W]
        piece_positions: New positions of each piece (for reference)
    """
    B, C, H, W = image_tensor.shape
    cell_h, cell_w = H // grid_size, W // grid_size
    
    # Create output tensor
    shuffled = image_tensor.clone()
    
    # Track piece positions (for reference)
    positions = list(range(grid_size * grid_size))
    
    # How many pieces to shuffle
    num_to_shuffle = int(len(positions) * shuffle_ratio)
    indices_to_shuffle = random.sample(positions, num_to_shuffle)
    
    # Randomly shuffle the selected positions
    shuffle_targets = indices_to_shuffle.copy()
    random.shuffle(shuffle_targets)
    
    # Create a mapping of original positions to new positions
    position_map = {i: i for i in range(grid_size * grid_size)}
    for i, j in zip(indices_to_shuffle, shuffle_targets):
        position_map[i] = j
    
    # Execute the shuffle
    temp_img = shuffled.clone()
    
    for orig_idx in range(grid_size * grid_size):
        # Calculate grid positions
        new_idx = position_map[orig_idx]
        
        # Original position
        orig_row, orig_col = orig_idx // grid_size, orig_idx % grid_size
        orig_y, orig_x = orig_row * cell_h, orig_col * cell_w
        
        # New position
        new_row, new_col = new_idx // grid_size, new_idx % grid_size
        new_y, new_x = new_row * cell_h, new_col * cell_w
        
        # Move the piece
        shuffled[:, :, new_y:new_y+cell_h, new_x:new_x+cell_w] = temp_img[:, :, orig_y:orig_y+cell_h, orig_x:orig_x+cell_w]
    
    return shuffled, position_map

# Load model
print("Loading model...")
model = PuzzleSolver(
    img_size=192,
    patch_size=16,
    grid_size=12,
    num_classes=144,
    embed_dim=192,
    depth=3,
    num_heads=3,
    mlp_ratio=4.0
)

checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
model.load_state_dict(checkpoint['model'])
model.to(device)
model.eval()
print("Model loaded successfully")

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((192, 192)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load test image
print("Loading test image:", test_image_path)
image = Image.open(test_image_path).convert('RGB')
original_tensor = transform(image).unsqueeze(0).to(device)
print(f"Original image size: {original_tensor.shape}")

# Create truly shuffled image
# Set random seed for reproducibility
random.seed(42)
shuffled_image, position_mapping = create_shuffled_image(
    original_tensor, 
    grid_size=12,
    shuffle_ratio=0.5  # Shuffle 50% of the pieces
)
print(f"Shuffled image size: {shuffled_image.shape}")
print(f"Shuffled {int(0.5 * 12 * 12)} pieces out of {12 * 12} total pieces")

# Basic visualization function
def visualize_tensor(tensor, title="Image", normalize=True):
    """Properly handle tensors of different shapes and visualize them"""
    # Ensure it's a CPU tensor
    if tensor.device.type != 'cpu':
        tensor = tensor.cpu()
    
    # Get tensor dimensions
    if len(tensor.shape) == 4:  # [B,C,H,W]
        tensor = tensor[0]  # Take the first sample
    
    C, H, W = tensor.shape
    
    if C == 3:  # Standard RGB image
        # Convert to [H,W,C]
        img = tensor.permute(1, 2, 0).numpy()
        if normalize:
            # Standard image denormalization
            img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img = np.clip(img, 0, 1)
    else:  # High-dimensional feature map - use first 3 channels
        # Method 1: Use first 3 channels
        if C >= 3:
            img = tensor[:3].permute(1, 2, 0).numpy()
        else:
            # If fewer than 3 channels, duplicate the last one
            channels = [tensor[i] for i in range(C)]
            while len(channels) < 3:
                channels.append(channels[-1])
            img = torch.stack(channels).permute(1, 2, 0).numpy()
        
        # Normalize to [0,1] range
        img = (img - img.min()) / (img.max() - img.min() + 1e-6)
    
    plt.imshow(img)
    plt.title(title)
    plt.axis('off')

# Enhanced multi-channel visualization function
def visualize_multi_channel(tensor, output_path, title="Feature Map Visualization"):
    """Create multiple different visualizations of high-dimensional feature maps"""
    if tensor.device.type != 'cpu':
        tensor = tensor.cpu()
    
    if len(tensor.shape) == 4:
        tensor = tensor[0]  # Take the first sample
    
    C, H, W = tensor.shape
    
    # Create 4 different visualizations
    plt.figure(figsize=(20, 15))
    
    # 1. First 3 channels as RGB
    plt.subplot(2, 2, 1)
    if C >= 3:
        img1 = tensor[:3].permute(1, 2, 0).numpy()
    else:
        channels = [tensor[i] for i in range(C)]
        while len(channels) < 3:
            channels.append(channels[-1])
        img1 = torch.stack(channels).permute(1, 2, 0).numpy()
    
    img1 = (img1 - img1.min()) / (img1.max() - img1.min() + 1e-6)
    plt.imshow(img1)
    plt.title("First 3 Channels as RGB")
    plt.axis('off')
    
    # 2. Average every 64 channels
    plt.subplot(2, 2, 2)
    num_groups = min(3, C // 1)
    grouped_channels = []
    
    for i in range(num_groups):
        start_idx = i * (C // num_groups)
        end_idx = (i + 1) * (C // num_groups) if i < num_groups - 1 else C
        grouped_channels.append(tensor[start_idx:end_idx].mean(dim=0))
    
    while len(grouped_channels) < 3:
        grouped_channels.append(grouped_channels[-1])
    
    img2 = torch.stack(grouped_channels).permute(1, 2, 0).numpy()
    img2 = (img2 - img2.min()) / (img2.max() - img2.min() + 1e-6)
    plt.imshow(img2)
    plt.title("Channel Group Averages")
    plt.axis('off')
    
    # 3. Use mean, std and max of channels
    plt.subplot(2, 2, 3)
    mean_channel = tensor.mean(dim=0, keepdim=True)
    std_channel = tensor.std(dim=0, keepdim=True)
    max_channel = tensor.max(dim=0, keepdim=True)[0]
    
    stat_channels = torch.cat([mean_channel, std_channel, max_channel], dim=0)
    img3 = stat_channels.permute(1, 2, 0).numpy()
    img3 = (img3 - img3.min()) / (img3.max() - img3.min() + 1e-6)
    plt.imshow(img3)
    plt.title("Mean/StdDev/Max")
    plt.axis('off')
    
    # 4. Show high-variance channels
    plt.subplot(2, 2, 4)
    # Calculate variance for each channel, select top 3 (most informative)
    var_per_channel = tensor.var(dim=(1, 2))
    top_channels = torch.topk(var_per_channel, min(3, C)).indices
    
    feature_channels = []
    for idx in top_channels:
        feature_channels.append(tensor[idx])
    
    while len(feature_channels) < 3:
        feature_channels.append(feature_channels[-1])
    
    img4 = torch.stack(feature_channels).permute(1, 2, 0).numpy()
    img4 = (img4 - img4.min()) / (img4.max() - img4.min() + 1e-6)
    plt.imshow(img4)
    plt.title("High Variance Channels")
    plt.axis('off')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Multi-channel visualization saved to {output_path}")

# Run model inference
print("Running model inference...")
with torch.no_grad():
    # Pass image directly to model
    position_logits, relation_logits, reconstructed_image = model(shuffled_image)
    
    print(f"Position prediction shape: {position_logits.shape}")
    print(f"Relation prediction shape: {relation_logits.shape}")
    print(f"Reconstructed image shape: {reconstructed_image.shape}")
    print(f"Reconstructed image min value: {reconstructed_image.min().item()}")
    print(f"Reconstructed image max value: {reconstructed_image.max().item()}")
    print(f"Reconstructed image mean value: {reconstructed_image.mean().item()}")

# 在测试脚本中添加解码器
decoder = FeatureDecoder(in_channels=192, out_channels=3).to(device)

# 手动初始化解码器权重（因为没有预训练权重）
def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)

decoder.apply(init_weights)

# 解码重建特征
with torch.no_grad():
    decoded_image = decoder(reconstructed_image)
    print(f"Decoded image shape: {decoded_image.shape}")

# 可视化解码后的图像
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
visualize_tensor(original_tensor[0], title="Original")
plt.subplot(1, 3, 2)
visualize_tensor(shuffled_image[0], title="Shuffled")
plt.subplot(1, 3, 3)
visualize_tensor(decoded_image[0], title="Decoded")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'decoded_comparison.png'), dpi=300)

# Save advanced feature map visualization
print("Generating advanced visualization...")
visualize_multi_channel(
    reconstructed_image[0], 
    os.path.join(output_dir, 'advanced_visualization.png'),
    title="Multiple Visualizations of Reconstructed Features"
)

print("Visualization complete, all results saved")