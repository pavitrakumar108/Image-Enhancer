import os
import argparse
import yaml
import torch

from src.data.df2k_loader import make_loader
from src.models.esrt_max import ESRT_MAX
from .trainer import Trainer


def pick_device():
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_cfg", default="configs/model/esrt_max_x4.yaml")
    ap.add_argument("--train_cfg", default="configs/training/train_x4.yaml")
    ap.add_argument("--data_cfg", default="configs/datasets/df2k_paths.yaml")
    args = ap.parse_args()

    # Load configs
    with open(args.model_cfg) as f:
        mcfg = yaml.safe_load(f)
    with open(args.train_cfg) as f:
        tcfg = yaml.safe_load(f)
    with open(args.data_cfg) as f:
        dcfg = yaml.safe_load(f)

    device = pick_device()

    # Model
    model = ESRT_MAX(
        scale=mcfg.get("scale", 4),
        c=mcfg.get("channels", 64),
        num_hpb=mcfg.get("num_hpb", 5),
        et_blocks=mcfg.get("transformer_blocks", 2),
        heads=mcfg.get("heads", 12),
        split=mcfg.get("split", 2)
    )

    # Dataloader settings
    patch = tcfg["patch_size"]
    batch = tcfg["batch_size"]
    workers = tcfg["workers"]
    degrade_mix = tcfg.get("degrade_mix", True)

    train_loader = make_loader(
        dcfg["train_roots"],
        mcfg["scale"],
        "train",
        patch,
        batch,
        workers,
        degrade_mix,
    )

    val_loader = make_loader(
        dcfg["train_roots"],
        mcfg["scale"],
        "val",
        patch,
        1,
        workers,
        False,
    )

    loaders = {"train": train_loader, "val": val_loader}

    # Steps
    steps_per_epoch = tcfg["steps_per_epoch"]
    epochs = tcfg["epochs"]
    total_steps = steps_per_epoch * epochs

    # Loss weights
    loss_cfg = tcfg.get("loss", {})
    w_ssim = float(loss_cfg.get("ssim", 0.05))
    w_edge = float(loss_cfg.get("edge", 0.02))

    trainer = Trainer(
        model=model,
        loaders=loaders,
        device=device,
        save_dir=tcfg["save_dir"],
        lr=tcfg["lr_initial"],
        warmup_steps=tcfg["warmup_steps"],
        total_steps=total_steps,
        ema_decay=tcfg["ema_decay"],
        accumulate_steps=tcfg["accumulate_steps"],
        w_ssim=w_ssim,
        w_edge=w_edge,
    )

    # Training loop
    best_psnr = 0

    for epoch in range(1, epochs + 1):
        trainer.train_epoch(epoch, steps_per_epoch)
        psnr = trainer.validate()
        print(f"[Epoch {epoch}] PSNR(EMA): {psnr:.4f} dB")

        if psnr > best_psnr:
            best_psnr = psnr
            trainer.save_ckpt("best.pth", {"psnr": psnr})

        if epoch % 20 == 0:
            trainer.save_ckpt(f"epoch_{epoch}.pth", {"psnr": psnr})


if __name__ == "__main__":
    main()
