import torch

class AccuracyMetrics:
    def __init__(self):
        pass

    def compute_accuracy(self, outputs, labels):
        _, preds = torch.max(outputs, 1)
        correct = (preds == labels).sum().item()
        total = labels.size(0)
        accuracy = correct / total
        return accuracy

    def compute_top_k_accuracy(self, outputs, labels, k=5):
        _, preds = torch.topk(outputs, k, dim=1)
        correct = preds.eq(labels.view(-1, 1).expand_as(preds)).sum().item()
        total = labels.size(0)
        top_k_accuracy = correct / total
        return top_k_accuracy
