import torch, os

def save_ckpt(state, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)

def load_ckpt(path, map_location='cpu'):
    return torch.load(path, map_location=map_location)
