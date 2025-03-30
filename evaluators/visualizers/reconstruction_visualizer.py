import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms

class ReconstructionVisualizer:
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        """
        Initialize the visualizer with optional normalization parameters
        mean and std: Normalization parameters used in the dataset
        """
        self.mean = mean
        self.std = std
    
    def _prepare_image(self, image):
        """Convert tensor to displayable numpy array with proper denormalization"""
        if isinstance(image, torch.Tensor):
            # Make a copy to avoid modifying the original
            image = image.clone().detach()
            
            # Handle batch dimension
            if image.dim() == 4:  # [B, C, H, W]
                image = image[0]  # Take first image in batch
            
            # Ensure channel dimension is correct
            if image.dim() == 3:  # [C, H, W]
                # Denormalize if needed
                if self.mean is not None and self.std is not None:
                    for t, m, s in zip(image, self.mean, self.std):
                        t.mul_(s).add_(m)
                
                # Clamp values to valid range [0, 1]
                image = torch.clamp(image, 0, 1)
                
                # Convert from [C, H, W] to [H, W, C] for matplotlib
                image = image.permute(1, 2, 0).cpu().numpy()
            else:
                # Handle grayscale or other formats
                image = image.cpu().numpy()
        
        # If already numpy array or other compatible format, return as is
        return image

    def visualize_reconstruction(self, original_image, shuffled_image, reconstructed_image, 
                                grid_size=4, save_path=None):
        """Visualize original, shuffled and reconstructed images side by side"""
        # Prepare images for visualization
        original_image = self._prepare_image(original_image)
        shuffled_image = self._prepare_image(shuffled_image)
        reconstructed_image = self._prepare_image(reconstructed_image)
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Display original image
        axes[0].imshow(original_image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        self._add_grid(axes[0], original_image, grid_size)
        
        # Display shuffled image
        axes[1].imshow(shuffled_image)
        axes[1].set_title('Shuffled Image')
        axes[1].axis('off')
        self._add_grid(axes[1], shuffled_image, grid_size)
        
        # Display reconstructed image
        axes[2].imshow(reconstructed_image)
        axes[2].set_title('Reconstructed Image')
        axes[2].axis('off')
        self._add_grid(axes[2], reconstructed_image, grid_size)
        
        plt.tight_layout()
        
        # Save figure if path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        plt.show()
    
    def _add_grid(self, ax, image, grid_size):
        """Add grid lines to show puzzle piece boundaries"""
        h, w = image.shape[0], image.shape[1]
        
        # Calculate grid cell size
        cell_h, cell_w = h / grid_size, w / grid_size
        
        # Draw horizontal lines
        for i in range(1, grid_size):
            y = i * cell_h
            ax.axhline(y, color='white', linewidth=1, alpha=0.7)
            
        # Draw vertical lines
        for i in range(1, grid_size):
            x = i * cell_w
            ax.axvline(x, color='white', linewidth=1, alpha=0.7)
