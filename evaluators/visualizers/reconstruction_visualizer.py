import matplotlib.pyplot as plt
import numpy as np

class ReconstructionVisualizer:
    def __init__(self):
        pass

    def visualize_reconstruction(self, original_image, shuffled_image, reconstructed_image):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(original_image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        axes[1].imshow(shuffled_image)
        axes[1].set_title('Shuffled Image')
        axes[1].axis('off')

        axes[2].imshow(reconstructed_image)
        axes[2].set_title('Reconstructed Image')
        axes[2].axis('off')

        plt.show()
