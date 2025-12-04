import os
import glob
import random
from typing import List, Tuple
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

from .transforms import random_flip_rot
from .degradations import rnd_degradation


######################################
# Helper functions
######################################

def read_rgb(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.asarray(img)


def to_tensor(img: np.ndarray) -> torch.Tensor:
    # Ensure writable array to avoid PyTorch warning
    img = img.copy()
    return torch.from_numpy(img.transpose(2, 0, 1)).float().div(255.0)


def gather_pairs(root: str, scale: int) -> List[Tuple[str, str]]:
    hr_dir = os.path.join(root, "HR")
    lr_dir = os.path.join(root, "LR_bicubic", f"X{scale}")

    pairs = []
    for hrp in sorted(glob.glob(os.path.join(hr_dir, "*.png"))):
        base = os.path.splitext(os.path.basename(hrp))[0]
        lrp = os.path.join(lr_dir, f"{base}x{scale}.png")
        if os.path.exists(lrp):
            pairs.append((hrp, lrp))
    return pairs


######################################
# DF2K Dataset
######################################

class DF2KDataset(Dataset):
    def __init__(self, roots, scale=4, split="train",
                 patch=96, degrade_mix=False, augment=True, seed=123):

        super().__init__()

        if isinstance(roots, str):
            roots = [roots]

        self.scale = scale
        self.patch = patch
        self.augment = augment
        self.degrade = degrade_mix

        # Build pairs
        pairs = []
        for r in roots:
            pairs.extend(gather_pairs(r, scale))

        if not pairs:
            raise FileNotFoundError(f"No HR/LR pairs found under {roots} for X{scale}")

        # Train/val split
        random.Random(seed).shuffle(pairs)
        cut = int(0.9 * len(pairs))
        self.pairs = pairs[:cut] if split == "train" else pairs[cut:]

    def __len__(self):
        return len(self.pairs)

    ######################################
    # Random crop for patches
    ######################################
    def _random_crop_pair(self, lr, hr):
        s = self.scale
        H, W = lr.shape[:2]
        ps = self.patch

        # pad if small
        if H < ps or W < ps:
            pad_h = max(0, ps - H)
            pad_w = max(0, ps - W)
            lr = np.pad(lr, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
            hr = np.pad(hr, ((0, pad_h * s), (0, pad_w * s), (0, 0)), mode="edge")
            H, W = lr.shape[:2]

        x = random.randint(0, W - ps)
        y = random.randint(0, H - ps)

        lr_crop = lr[y:y + ps, x:x + ps]
        hr_crop = hr[y * s:(y + ps) * s, x * s:(x + ps) * s]

        return lr_crop, hr_crop

    ######################################
    # __getitem__
    ######################################
    def __getitem__(self, idx):
        hr_path, lr_path = self.pairs[idx]

        hr = read_rgb(hr_path)

        if self.degrade:
            lr = rnd_degradation(hr, self.scale)
        else:
            lr = read_rgb(lr_path)

        if self.augment:
            lr, hr = random_flip_rot(lr, hr)

        lr, hr = self._random_crop_pair(lr, hr)

        return to_tensor(lr), to_tensor(hr)


######################################
# Make DataLoader (NO WARNINGS)
######################################

def make_loader(roots, scale, split, patch, batch, workers,
                degrade_mix=False, shuffle=True):

    dataset = DF2KDataset(
        roots=roots,
        scale=scale,
        split=split,
        patch=patch,
        degrade_mix=degrade_mix,
        augment=(split == "train")
    )

    # detect device
    if torch.cuda.is_available():
        pin = True
    elif torch.backends.mps.is_available():
        pin = False  # MPS does NOT support pin_memory
    else:
        pin = False

    return DataLoader(
        dataset,
        batch_size=batch,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=pin,
        drop_last=(split == "train")
    )
