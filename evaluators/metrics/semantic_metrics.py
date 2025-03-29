import torch
import torch.nn.functional as F
from torchvision import models, transforms

class SemanticMetrics:
    def __init__(self):
        self.model = models.resnet50(pretrained=True)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def compute_feature_similarity(self, img1, img2):
        img1 = self.transform(img1).unsqueeze(0)
        img2 = self.transform(img2).unsqueeze(0)

        with torch.no_grad():
            features1 = self.model(img1)
            features2 = self.model(img2)

        similarity = F.cosine_similarity(features1, features2).item()
        return similarity

    def compute_semantic_consistency(self, img1, img2):
        similarity = self.compute_feature_similarity(img1, img2)
        return similarity
