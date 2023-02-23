import torch
import torch.nn as nn
import torch.nn.functional as func
from src.metrics import psnr,ssim

class SSIMLoss(nn.Module):  #ssim loss
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2):
        return 1-ssim(x1, x2)

