import torch
import torch.nn as nn
from models.puzzle_solver import PuzzleSolver

class Predictor:
    def __init__(self, model_path, device='cpu'):
        self.device = torch.device(device)
        self.model = PuzzleSolver()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def predict(self, image_patches):
        with torch.no_grad():
            image_patches = image_patches.to(self.device)
            position_logits, relation_logits, reconstructed_image = self.model(image_patches)
        return position_logits, relation_logits, reconstructed_image
