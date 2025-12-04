#!/usr/bin/env bash
set -e
python -m src.training.train_pipeline \
  --model_cfg configs/model/esrt_max_x4.yaml \
  --train_cfg configs/training/train_x4.yaml \
  --data_cfg  configs/datasets/df2k_paths.yaml
