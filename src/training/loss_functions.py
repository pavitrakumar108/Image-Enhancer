import torch
import torch.nn as nn
import torch.nn.functional as F


###################################
# --- Y CHANNEL CONVERSION ---
###################################
def to_y(img):
    r, g, b = img[:, 0:1], img[:, 1:2], img[:, 2:3]
    return 0.299 * r + 0.587 * g + 0.114 * b


###################################
# --- SSIM LOSS (AMP-FRIENDLY) ---
###################################
@torch.no_grad()
def ssim_map(x, y, C1=0.01**2, C2=0.03**2):
    # Use avg_pool2d for AMP-safe smoothing
    mu_x = F.avg_pool2d(x, 11, stride=1, padding=5)
    mu_y = F.avg_pool2d(y, 11, stride=1, padding=5)

    sigma_x  = F.avg_pool2d(x * x, 11, stride=1, padding=5) - mu_x**2
    sigma_y  = F.avg_pool2d(y * y, 11, stride=1, padding=5) - mu_y**2
    sigma_xy = F.avg_pool2d(x * y, 11, stride=1, padding=5) - mu_x * mu_y

    num = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    den = (mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2)

    return num / (den + 1e-12)


def ssim_loss(sr, hr):
    # Compute on Y-channel, more stable under AMP
    y1, y2 = to_y(sr), to_y(hr)
    return 1.0 - ssim_map(y1, y2).mean()


###################################
# --- EDGE LOSS (STABLE + FAST) ---
###################################
def sobel_edges(img):
    # Sobel kernels (pre-allocated for AMP safety)
    device, dtype = img.device, img.dtype

    kx = torch.tensor([[1, 0, -1],
                       [2, 0, -2],
                       [1, 0, -1]], device=device, dtype=dtype).view(1,1,3,3)

    ky = torch.tensor([[1,  2,  1],
                       [0,  0,  0],
                       [-1, -2, -1]], device=device, dtype=dtype).view(1,1,3,3)

    y = to_y(img)
    gx = F.conv2d(y, kx, padding=1)
    gy = F.conv2d(y, ky, padding=1)

    # Avoid sqrt instability
    return torch.sqrt(gx * gx + gy * gy + 1e-6)


def edge_loss(sr, hr):
    return (sobel_edges(sr) - sobel_edges(hr)).abs().mean()


###################################
# --- FINAL LOSS (L1 + SSIM + EDGE)
###################################
class FinalLoss(nn.Module):
    def __init__(self, w_ssim=0.05, w_edge=0.02):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.w_ssim = float(w_ssim)
        self.w_edge = float(w_edge)

    def forward(self, sr, hr):
        l_l1 = self.l1(sr, hr)
        l_ssim = ssim_loss(sr, hr)
        l_edge = edge_loss(sr, hr)
        return l_l1 + self.w_ssim * l_ssim + self.w_edge * l_edge


###################################
# --- PYRAMID LOSS (x2 + x4)
###################################
class PyramidLoss(nn.Module):
    def __init__(self, w_ssim=0.05, w_edge=0.02):
        super().__init__()
        self.base = FinalLoss(w_ssim=w_ssim, w_edge=w_edge)

    def forward(self, y2, y4, hr2, hr4):
        # High-resolution branch prioritised
        return self.base(y4, hr4) + 0.5 * self.base(y2, hr2)
