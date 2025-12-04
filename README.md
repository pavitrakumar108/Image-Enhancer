Image-Enhancer – Image Super-Resolution using Efficient Transformers (ESRT)

This repository contains an implementation of the Efficient Super-Resolution Transformer (ESRT) model for high-quality image enhancement and super-resolution. The project focuses on reconstructing high-resolution images from low-resolution inputs using transformer-based architectures optimized for speed, memory efficiency, and real-world performance.

Image-Enhancer currently supports ESRT, and the structure is designed to easily plug in future models like ESRGAN, SwinIR, NAFNet, and Restormer.

Overview

Traditional image enhancement and super-resolution techniques often struggle with
✔️ real-world degradations
✔️ low-light noise
✔️ motion blur
✔️ edge preservation
✔️ computational cost on consumer hardware

Recent transformer-based architectures have shown strong performance in super-resolution, but they often require heavy compute. ESRT solves this by combining:

lightweight transformers

CNN-based residual blocks

hierarchical feature extraction

kernel-aware processing

This project implements the full ESRT pipeline — including model architecture, dataloaders, training loops, evaluation tools, and inference scripts — in a structured and modular way that makes experimentation easy.

The long-term goal is to provide a clean benchmarking hub for comparing different super-resolution architectures under the same training setup.

Key Features

🔥 Complete ESRT implementation (training + validation + inference)

📁 Clean modular code structure for easy experimentation

📈 Supports PSNR/SSIM evaluation

🧪 Experiment tracking and checkpoints (via Git LFS)

🧰 Tools for dataset preparation and visualization

🚀 Designed to add more models later (ESRGAN, SwinIR, NAFNet, Restormer)

Dataset

This project uses DF2K, a high-quality super-resolution dataset consisting of:

📦 DIV2K

📦 Flickr2K

📦 OST (Outdoor Scenes Dataset)

The dataset is stored locally inside:

datasets/DF2K/


Note:
The dataset is not uploaded to GitHub (ignored via .gitignore).
You must download DF2K manually before training.

Applications

This project demonstrates techniques used in:

📸 Photography enhancement (mobile, DSLR, CCTV images)

🩺 Medical image upscaling (X-ray, MRI pre-processing)

🛰️ Satellite and aerial imagery restoration

🕵️‍♂️ Forensic image enhancement

🎥 Video upscaling (future extension)

Super-resolution is widely used when original high-quality data is not available and restoring details is essential.

📁 File Structure
Image-Enhancer/
│
├── datasets/                           # (Ignored in GitHub – local only)
│   └── DF2K/                           # DIV2K + Flickr2K + OST datasets
│
├── models/
│   └── esrt/                           # ESRT model (main focus of project)
│       ├── src/
│       │   ├── data/                   # Dataloaders, degradations, transforms
│       │   ├── models/                 # ESRT model & building blocks
│       │   ├── training/               # Trainer, loss, scheduler, EMA, metrics
│       │   ├── evaluation/             # PSNR/SSIM, single-image inference, TTA
│       │   └── utils/                  # Logger, checkpoint IO, visualization
│       │
│       ├── configs/                    # YAML configs (model + runtime + training)
│       │   ├── model/
│       │   │   └── esrt_max_x4.yaml    # ESRT architecture config
│       │   ├── training/
│       │   │   └── train_x4.yaml       # Training config
│       │   └── runtime.yaml            # General settings
│       │
│       ├── experiments/                # Trained model checkpoints (Git-LFS)
│       │   ├── esrt_fast_best/
│       │   │   └── checkpoints/        # best.pth, epoch_xx.pth
│       │   └── exp1_esrt_max_x4/       # New training runs saved here
│       │
│       ├── results/                    # Outputs from inference & evaluation
│       │   └── (PSNR_SSIM_reports + images)
│       │
│       ├── scripts/                    # Automation scripts
│       │   ├── train.sh                # Start training
│       │   ├── validate.sh             # Compute PSNR/SSIM
│       │   ├── infer.sh                # Inference on images
│       │   └── prepare_dataset.sh      # Dataset setup
│       │
│       ├── tools/                      # Helper scripts
│       │   ├── visualize_pairs.py
│       │   ├── auto_generate_lr.py
│       │   └── check_dataset.py
│       │
│       └── README.md                   # ESRT-specific documentation
│
├── docs/                               # Architecture diagrams, research notes
│   └── ESRT.pdf                        # ESRT architecture (your uploaded PDF)
│
├── .gitattributes                      # Git-LFS tracking rules
├── .gitignore                          # Ignoring datasets & temp files
└── README.md                           # Global project documentions
