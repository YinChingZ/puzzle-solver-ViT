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

def calculate_position_accuracy(position_logits, target_positions):
    """
    计算位置预测准确率

    参数:
        position_logits: 位置预测的logits
        target_positions: 真实位置标签

    返回:
        position_accuracy: 位置预测准确率
    """
    _, position_preds = torch.max(position_logits, 1)
    correct = (position_preds == target_positions).sum().item()
    total = target_positions.size(0)
    position_accuracy = correct / total
    return position_accuracy

def calculate_relationship_accuracy(relation_logits, target_relations):
    """
    计算关系预测准确率

    参数:
        relation_logits: 关系预测的logits
        target_relations: 真实关系标签

    返回:
        relationship_accuracy: 关系预测准确率
    """
    _, relation_preds = torch.max(relation_logits, 1)
    correct = (relation_preds == target_relations).sum().item()
    total = target_relations.size(0)
    relationship_accuracy = correct / total
    return relationship_accuracy
