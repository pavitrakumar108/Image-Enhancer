import math, torch
import torch.nn as nn
import torch.nn.functional as F
from .film import FiLM

def conv3x3(in_ch,out_ch): return nn.Conv2d(in_ch,out_ch,3,padding=1)

class PixelShuffleUpsampler(nn.Module):
    def __init__(self, scale, c):
        super().__init__()
        body=[]
        if scale in (2,4):
            for _ in range(int(math.log(scale,2))):
                body += [conv3x3(c,4*c), nn.PixelShuffle(2), nn.GELU()]
        elif scale==3:
            body += [conv3x3(c,9*c), nn.PixelShuffle(3), nn.GELU()]
        self.body=nn.Sequential(*body)
    def forward(self,x): return self.body(x)

class HFM(nn.Module):
    def __init__(self, k=2): super().__init__(); self.k=k
    def forward(self,x):
        B,C,H,W = x.shape
        ta = F.avg_pool2d(x, kernel_size=self.k, stride=self.k)
        tu = F.interpolate(ta, size=(H,W), mode='bilinear', align_corners=False)
        return x - tu

class ResidualUnitRE(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.reduce = nn.Conv2d(c, c//2, 1)
        self.expand = nn.Conv2d(c//2, c, 1)
        self.lambda_res = nn.Parameter(torch.tensor(1.0))
        self.lambda_x   = nn.Parameter(torch.tensor(1.0))
        self.act = nn.GELU()
    def forward(self,x):
        y = self.expand(self.act(self.reduce(x)))
        return self.lambda_res*y + self.lambda_x*x

class ARFB(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ru1 = ResidualUnitRE(c)
        self.ru2 = ResidualUnitRE(c)
        self.fuse = nn.Conv2d(2*c, c, 1)
        self.conv3 = conv3x3(c, c)
        self.act = nn.GELU()
    def forward(self,x):
        y1 = self.ru1(x)
        y2 = self.ru2(y1)
        y  = torch.cat([y1,y2], dim=1)
        y  = self.fuse(y)
        y  = self.conv3(self.act(y))
        return y

class SEChannelAttention(nn.Module):
    def __init__(self, c, r=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(c, c//r, 1), nn.GELU(),
            nn.Conv2d(c//r, c, 1), nn.Sigmoid()
        )
    def forward(self,x): return x * self.fc(self.pool(x))

class HPB(nn.Module):
    def __init__(self, c, hfm_k=2, repeats=5, code_dim=64):
        super().__init__()
        self.pre = ARFB(c)
        self.hfm = HFM(k=hfm_k)
        self.down = nn.Conv2d(c, c, 3, stride=2, padding=1)
        self.shared = ARFB(c)
        self.repeats=repeats
        self.align = ARFB(c)
        self.fuse1x1 = nn.Conv2d(2*c, c, 1)
        self.ca = SEChannelAttention(c)
        self.post = ARFB(c)
        self.film = FiLM(c, code_dim=code_dim)

    def forward(self,x, code=None):
        x_in = x
        x1 = self.pre(x)
        if code is not None: x1 = self.film(x1, code)
        phigh = self.align(self.hfm(x1))
        x2 = self.down(x1)
        for _ in range(self.repeats):
            x2 = self.shared(x2)
        x2 = F.interpolate(x2, size=x1.shape[2:], mode='bilinear', align_corners=False)
        y = self.fuse1x1(torch.cat([phigh, x2], dim=1))
        y = self.ca(y)
        y = self.post(y)
        return y + x_in

class GateFuse(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(2*c, c, 3, padding=1), nn.GELU(),
            nn.Conv2d(c, 1, 3, padding=1), nn.Sigmoid()
        )
    def forward(self, fa, fb):
        g = self.fuse(torch.cat([fa, fb], dim=1))
        return g*fa + (1-g)*fb
