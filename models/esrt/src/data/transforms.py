import numpy as np, random

def random_flip_rot(lr: np.ndarray, hr: np.ndarray):
    if random.random() < 0.5:
        lr = lr[:, ::-1].copy(); hr = hr[:, ::-1].copy()
    if random.random() < 0.5:
        lr = lr[::-1, :].copy(); hr = hr[::-1, :].copy()
    if random.random() < 0.5:
        lr = np.rot90(lr).copy(); hr = np.rot90(hr).copy()
    return lr, hr
