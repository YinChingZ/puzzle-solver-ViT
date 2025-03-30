import torch
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def calculate_accuracy(logits, targets):
    """
    计算分类准确率

    参数:
        logits: 模型输出的logits
        targets: 真实标签

    返回:
        accuracy: 分类准确率
    """
    _, preds = torch.max(logits, 1)
    correct = (preds == targets).sum().item()
    total = targets.size(0)
    accuracy = correct / total
    return accuracy

def calculate_psnr(pred_img, target_img):
    """
    计算峰值信噪比 (PSNR)

    参数:
        pred_img: 预测图像
        target_img: 真实图像

    返回:
        psnr: 峰值信噪比
    """
    pred_img_np = pred_img.cpu().numpy()
    target_img_np = target_img.cpu().numpy()
    psnr = peak_signal_noise_ratio(target_img_np, pred_img_np)
    return psnr

def calculate_ssim(pred_img, target_img):
    """
    计算结构相似性 (SSIM)

    参数:
        pred_img: 预测图像
        target_img: 真实图像

    返回:
        ssim: 结构相似性
    """
    pred_img_np = pred_img.cpu().numpy()
    target_img_np = target_img.cpu().numpy()
    ssim = structural_similarity(target_img_np, pred_img_np, multichannel=True)
    return ssim
