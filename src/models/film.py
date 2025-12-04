import torch, torch.nn as nn

class FiLM(nn.Module):
    def __init__(self, c, code_dim=64):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(code_dim, 2*c), nn.GELU(), nn.Linear(2*c, 2*c))
    def forward(self, x, code):
        gamma_beta = self.fc(code)
        B,C,H,W = x.shape
        gamma, beta = gamma_beta[:, :C].view(B,C,1,1), gamma_beta[:, C:].view(B,C,1,1)
        return x * (1 + gamma) + beta
