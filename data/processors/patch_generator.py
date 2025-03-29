import numpy as np
from PIL import Image

class PatchGenerator:
    def __init__(self, grid_size):
        self.grid_size = grid_size

    def generate_patches(self, image):
        image = Image.open(image).convert('RGB')
        image = np.array(image)
        patches = self._create_patches(image)
        return patches

    def _create_patches(self, image):
        patches = []
        img_height, img_width, _ = image.shape
        patch_height = img_height // self.grid_size
        patch_width = img_width // self.grid_size

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                patch = image[i * patch_height:(i + 1) * patch_height, j * patch_width:(j + 1) * patch_width]
                patches.append(patch)

        return patches
