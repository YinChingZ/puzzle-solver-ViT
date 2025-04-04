import torch
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import lpips

class ImageQualityMetrics:
    def __init__(self):
        self.lpips_loss = lpips.LPIPS(net='alex')

    def compute_psnr(self, img1, img2):
        psnr = peak_signal_noise_ratio(img1, img2)
        return psnr

    def compute_ssim(self, img1, img2):
        ssim = structural_similarity(img1, img2, multichannel=True)
        return ssim

    def compute_lpips(self, img1, img2):
        img1 = torch.tensor(img1).permute(2, 0, 1).unsqueeze(0).float()
        img2 = torch.tensor(img2).permute(2, 0, 1).unsqueeze(0).float()
        lpips_value = self.lpips_loss(img1, img2).item()
        return lpips_value
