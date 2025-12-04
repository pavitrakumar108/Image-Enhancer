#!/usr/bin/env bash
set -e
CKPT=${1:-experiments/exp1_esrt_max_x4/checkpoints/best.pth}
python -m src.evaluation.evaluate_psnr_ssim --ckpt "$CKPT"
