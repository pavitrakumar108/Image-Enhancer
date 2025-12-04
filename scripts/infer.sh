#!/usr/bin/env bash
set -e
CKPT=${1}
INP=${2}
OUT=${3:-sr_out.png}
if [ -z "$CKPT" ] || [ -z "$INP" ]; then
  echo "Usage: bash scripts/infer.sh <ckpt.pth> <input.png> [output.png]"; exit 1
fi
python -m src.evaluation.infer_single_image --ckpt "$CKPT" --inp "$INP" --out "$OUT"
