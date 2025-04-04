import torch
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import warnings

def calculate_accuracy(logits, targets):
    """
    计算分类准确率
    
    参数:
        logits: 模型输出的预测分数 [B, C]
        targets: 真实标签 [B]
        
    返回:
        float: 准确率 (0.0-1.0)
    """
    _, preds = torch.max(logits, 1)
    correct = (preds == targets).sum().item()
    total = targets.size(0)
    return correct / total

def calculate_top_k_accuracy(logits, targets, k=5):
    """
    计算Top-K准确率
    
    参数:
        logits: 模型输出的预测分数 [B, C]
        targets: 真实标签 [B]
        k: 前k个预测
        
    返回:
        float: Top-K准确率 (0.0-1.0)
    """
    _, pred = logits.topk(k, 1, True, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))
    correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
    return correct_k.item() / targets.size(0)

def calculate_psnr(pred_img, target_img):
    """
    计算峰值信噪比 (PSNR)
    
    参数:
        pred_img: 预测图像 (张量或NumPy数组)
        target_img: 目标图像 (张量或NumPy数组)
        
    返回:
        float: PSNR值 (dB)
    """
    # 确保输入是NumPy数组
    if isinstance(pred_img, torch.Tensor):
        pred_img = pred_img.detach().cpu().numpy()
    if isinstance(target_img, torch.Tensor):
        target_img = target_img.detach().cpu().numpy()
    
    # 处理批次维度
    if pred_img.ndim == 4:  # [B,C,H,W]
        result = 0
        for i in range(pred_img.shape[0]):
            result += _calculate_psnr_single(pred_img[i], target_img[i])
        return result / pred_img.shape[0]
    else:
        return _calculate_psnr_single(pred_img, target_img)

def _calculate_psnr_single(pred_img, target_img):
    """单图像PSNR计算"""
    # 通道维度处理 - 从PyTorch格式 [C,H,W] 转换为 [H,W,C]
    if pred_img.shape[0] == 1 or pred_img.shape[0] == 3:  # 如果第一维是通道
        pred_img = np.transpose(pred_img, (1, 2, 0))
        target_img = np.transpose(target_img, (1, 2, 0))
    
    # 确保值在[0,1]范围内
    pred_img = np.clip(pred_img, 0, 1)
    target_img = np.clip(target_img, 0, 1)
    
    return peak_signal_noise_ratio(target_img, pred_img, data_range=1.0)

def calculate_ssim(pred_img, target_img):
    """
    计算结构相似性 (SSIM)
    
    参数:
        pred_img: 预测图像 (张量或NumPy数组)
        target_img: 目标图像 (张量或NumPy数组)
        
    返回:
        float: SSIM值 (0.0-1.0)
    """
    # 确保输入是NumPy数组
    if isinstance(pred_img, torch.Tensor):
        pred_img = pred_img.detach().cpu().numpy()
    if isinstance(target_img, torch.Tensor):
        target_img = target_img.detach().cpu().numpy()
    
    # 处理批次维度
    if pred_img.ndim == 4:  # [B,C,H,W]
        result = 0
        for i in range(pred_img.shape[0]):
            result += _calculate_ssim_single(pred_img[i], target_img[i])
        return result / pred_img.shape[0]
    else:
        return _calculate_ssim_single(pred_img, target_img)

def _calculate_ssim_single(pred_img, target_img):
    """单图像SSIM计算"""
    # 通道维度处理 - 从PyTorch格式 [C,H,W] 转换为 [H,W,C]
    if pred_img.shape[0] == 1 or pred_img.shape[0] == 3:  # 如果第一维是通道
        pred_img = np.transpose(pred_img, (1, 2, 0))
        target_img = np.transpose(target_img, (1, 2, 0))
    
    # 确保值在[0,1]范围内
    pred_img = np.clip(pred_img, 0, 1)
    target_img = np.clip(target_img, 0, 1)
    
    # 处理不同版本的scikit-image API
    try:
        # 较新版本使用 channel_axis
        return structural_similarity(target_img, pred_img, channel_axis=-1 if pred_img.ndim > 2 else None, data_range=1.0)
    except TypeError:
        # 兼容旧版本
        warnings.warn("Using deprecated 'multichannel' parameter in structural_similarity")
        return structural_similarity(target_img, pred_img, multichannel=pred_img.ndim > 2, data_range=1.0)
