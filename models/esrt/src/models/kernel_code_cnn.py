import torch, torch.nn as nn

class KernelCode(nn.Module):
    def __init__(self, code_dim=64):
        super().__init__()
        c=32
        self.body = nn.Sequential(
            nn.Conv2d(3, c, 3, padding=1), nn.GELU(),
            nn.Conv2d(c, c, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(c, 2*c, 3, stride=2, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(2*c, code_dim)
    def forward(self, x):
        f = self.body(x)
        f = f.view(f.size(0), -1)
        return self.fc(f)
