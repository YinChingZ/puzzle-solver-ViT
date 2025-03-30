import torch
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def calculate_accuracy(logits, targets):
    _, preds = torch.max(logits, 1)
    correct = (preds == targets).sum().item()
    total = targets.size(0)
    accuracy = correct / total
    return accuracy

def calculate_psnr(pred_img, target_img):
    psnr = peak_signal_noise_ratio(target_img, pred_img)
    return psnr

def calculate_ssim(pred_img, target_img):
    ssim = structural_similarity(target_img, pred_img, multichannel=True)
    return ssim
