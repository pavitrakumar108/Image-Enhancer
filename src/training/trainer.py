import os
import torch
from torch.optim import Adam
from tqdm import tqdm
from .loss_functions import PyramidLoss, to_y
from .ema import EMA
from .scheduler import build_cosine_with_warmup
from .metrics import psnr


def amp_device_type(device):
    if "cuda" in device:
        return "cuda"
    if device == "mps":
        return "mps"
    return "cpu"


class Trainer:
    def __init__(
        self,
        model,
        loaders,
        device,
        save_dir,
        lr,
        warmup_steps,
        total_steps,
        ema_decay,
        accumulate_steps,
        w_ssim,
        w_edge,
    ):

        self.device = device
        self.model = model.to(device)

        self.train_loader = loaders["train"]
        self.val_loader = loaders["val"]

        self.optimizer = Adam(self.model.parameters(), lr=lr, betas=(0.9, 0.999))
        self.scheduler = build_cosine_with_warmup(
            self.optimizer, total_steps, warmup_steps
        )

        self.criterion = PyramidLoss(w_ssim, w_edge)

        amp_dev = amp_device_type(device)
        self.scaler = torch.amp.GradScaler(
            amp_dev if amp_dev in ("cuda", "mps") else None
        )

        self.accumulate_steps = max(1, accumulate_steps)
        self.ema = EMA(self.model, decay=ema_decay)

        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def down2(self, hr):
        return torch.nn.functional.interpolate(
            hr, scale_factor=0.5, mode="bicubic", align_corners=False
        )

    def train_epoch(self, epoch, steps_per_epoch):
        self.model.train()

        # FIX 1: correct total steps shown in tqdm
        loop = tqdm(self.train_loader, desc=f"Epoch {epoch}", total=steps_per_epoch)

        step = 0
        self.optimizer.zero_grad(set_to_none=True)

        device = self.device
        amp_dev = amp_device_type(device)

        for lr, hr in loop:
            lr = lr.to(device, non_blocking=True)
            hr = hr.to(device, non_blocking=True)
            hr2 = self.down2(hr)

            with torch.amp.autocast(device_type=amp_dev):
                y2, y4 = self.model(lr)
                loss = self.criterion(
                    y2.clamp(0, 1), y4.clamp(0, 1), hr2, hr
                )
                loss = loss / self.accumulate_steps

            self.scaler.scale(loss).backward()

            if (step + 1) % self.accumulate_steps == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.ema.update(self.model)
                self.scheduler.step()

            # FIX 2: remove scalar warning with detach()
            loop.set_postfix(
                loss=f"{float((loss * self.accumulate_steps).detach()):.4f}"
            )

            step += 1
            if step >= steps_per_epoch:
                break

        # flush remaining grads if any
        if step % self.accumulate_steps != 0:
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            self.ema.update(self.model)

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        ema_model = self.ema.ema
        vals = []
        for lr, hr in self.val_loader:
            lr = lr.to(self.device, non_blocking=True)
            hr = hr.to(self.device, non_blocking=True)
            _, y4 = ema_model(lr)
            y4 = y4.clamp(0, 1)
            vals.append(psnr(to_y(y4), to_y(hr)).item())
        return sum(vals) / len(vals)

    def save_ckpt(self, name, extra=None):
        sd = {"ema": self.ema.state_dict()}
        if extra:
            sd.update(extra)
        torch.save(sd, os.path.join(self.save_dir, name))
