import numpy as np

class Shuffler:
    def __init__(self, difficulty):
        self.difficulty = difficulty

    def shuffle_patches(self, patches):
        if self.difficulty == 'easy':
            return self._neighbor_swap(patches)
        elif self.difficulty == 'medium':
            return self._local_shuffle(patches)
        elif self.difficulty == 'hard':
            return self._global_shuffle(patches)
        else:
            raise ValueError("Invalid difficulty level")

    def _neighbor_swap(self, patches):
        shuffled_patches = patches.copy()
        for i in range(len(patches)):
            if i % 2 == 0 and i + 1 < len(patches):
                shuffled_patches[i], shuffled_patches[i + 1] = shuffled_patches[i + 1], shuffled_patches[i]
        return shuffled_patches

    def _local_shuffle(self, patches):
        shuffled_patches = patches.copy()
        np.random.shuffle(shuffled_patches)
        return shuffled_patches

    def _global_shuffle(self, patches):
        shuffled_patches = patches.copy()
        np.random.shuffle(shuffled_patches)
        return shuffled_patches
