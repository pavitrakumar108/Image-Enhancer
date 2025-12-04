import os, glob
from PIL import Image
import numpy as np
import cv2

def load_image(path):
    img = Image.open(path).convert("RGB")
    return np.array(img)

def save_image(path, img):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(img).save(path)

def list_images(folder):
    exts = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp"]
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(folder, "**", ext), recursive=True))
    return sorted(files)

def mod_crop(img, scale):
    h, w = img.shape[:2]
    h_r = (h // scale) * scale
    w_r = (w // scale) * scale
    return img[:h_r, :w_r, :]

def resize_bicubic(img, scale):
    h, w = img.shape[:2]
    return cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)

def numpy_to_tensor(img):
    import torch
    return torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0

def validate_sr_dataset(root, scale):
    hr_dir = os.path.join(root, "HR")
    lr_dir = os.path.join(root, "LR_bicubic", f"X{scale}")
    if not os.path.exists(hr_dir):
        raise FileNotFoundError(f"Missing folder: {hr_dir}")
    if not os.path.exists(lr_dir):
        raise FileNotFoundError(f"Missing folder: {lr_dir}")
    hr_files = list_images(hr_dir)
    lr_files = list_images(lr_dir)
    if len(hr_files) == 0:
        raise ValueError("HR folder is empty.")
    if len(lr_files) == 0:
        raise ValueError("LR folder is empty.")
    return True

def match_lr_hr_names(lr_path, hr_path, scale):
    lr_base = os.path.splitext(os.path.basename(lr_path))[0]
    hr_base = os.path.splitext(os.path.basename(hr_path))[0]
    expected_lr = hr_base + f"x{scale}"
    return lr_base == expected_lr
