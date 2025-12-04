<h1 align="center">📸 Image-Enhancer – Efficient Super-Resolution using ESRT</h1>
<p align="center">
  High-quality image enhancement using Efficient Super-Resolution Transformers (ESRT)
</p>
<p align="center">
  <strong>Clean • Modular • Research-Friendly • High-Performance</strong>
</p>
<br/>

<h2>📘 About the Project</h2>
<p>
This repository provides a complete implementation of the 
<strong>Efficient Super-Resolution Transformer (ESRT)</strong> for generating 
sharp, high-quality images from low-resolution inputs. 
It includes the full pipeline — model architecture, training framework, evaluation tools, 
and inference scripts — all organized for research and experimentation.
</p>

<p>
The project is structured to support additional models in the future 
(e.g., ESRGAN, SwinIR, NAFNet, Restormer) while keeping ESRT as the core implementation.
</p>

<hr/>

<h2>📌 Overview</h2>
<p>
Traditional super-resolution methods struggle with <strong>noise, blur, artifacts, 
low-light conditions, and edge preservation</strong>. Many deep models achieve high quality 
but require large computational resources.
</p>

<p>
<strong>ESRT solves these problems by combining:</strong>
</p>

<ul>
  <li>⚡ Lightweight Transformer blocks</li>
  <li>🧠 CNN-based hierarchical feature extraction</li>
  <li>🔗 Kernel-aware operations</li>
  <li>📉 Low memory usage + high efficiency</li>
</ul>

<p>
This repository provides a research-friendly implementation focusing on clarity, modularity, 
and real-world performance.
</p>

<hr/>

<h2>🔥 Key Features</h2>
<ul>
  <li>Complete ESRT pipeline — training, evaluation, and inference</li>
  <li>PSNR and SSIM evaluation support</li>
  <li>Clean, modular code structure for experimentation</li>
  <li>Dataset utilities for DF2K preparation</li>
  <li>Git-LFS for storing large model checkpoints</li>
  <li>Well-organized folder structure for future extensions</li>
</ul>

<hr/>

<h2>🗂️ Dataset</h2>
<p>
This project uses the <strong>DF2K</strong> dataset:
</p>

<ul>
  <li>DIV2K</li>
  <li>Flickr2K</li>
  <li>OST (Outdoor Scenes)</li>
</ul>

<p>Place the dataset locally:</p>

<pre>
<code>datasets/DF2K/</code>
</pre>

<p><em>The dataset is ignored in GitHub via <code>.gitignore</code>.</em></p>

<hr/>

<h2>🚀 Applications</h2>
<ul>
  <li>📸 Photography enhancement (DSLR, mobile, CCTV)</li>
  <li>🩺 Medical image improvement (X-Ray, CT, MRI)</li>
  <li>🛰️ Satellite & aerial image restoration</li>
  <li>🔍 Forensic image enhancement</li>
  <li>🎥 Video upscaling (future)</li>
</ul>

<hr/>

<h2>📁 Repository Structure</h2>

<pre>
<code>
Image-Enhancer/
│
├── datasets/                        # Local dataset (ignored in Git)
│   └── DF2K/
│
├── models/
│   └── esrt/
│       ├── src/
│       │   ├── data/                # Dataloaders, transforms, degradations
│       │   ├── models/              # ESRT model components
│       │   ├── training/            # Trainer, EMA, scheduler, losses
│       │   ├── evaluation/          # PSNR, SSIM, inference, TTA
│       │   └── utils/               # Logging, visualization, ckpt IO
│       │
│       ├── configs/                 # YAML configs
│       │   ├── model/
│       │   ├── training/
│       │   └── runtime.yaml
│       │
│       ├── experiments/             # Git-LFS checkpoints
│       ├── results/                 # Output images + metrics
│       ├── scripts/                 # Shell scripts (train, infer, validate)
│       └── tools/                   # Helper utilities
│
├── docs/
│   └── ESRT.pdf                     # Architecture document
│
├── .gitattributes                   # LFS rules
├── .gitignore                       # Ignore datasets/temp files
└── README.md                        # Global documentation
</code>
</pre>

<hr/>

<h2>⚡ Quick Start</h2>

<h3>Train</h3>
<pre><code>
cd models/esrt/scripts
bash train.sh
</code></pre>

<h3>Evaluate (PSNR / SSIM)</h3>
<pre><code>
bash validate.sh
</code></pre>

<h3>Inference</h3>
<pre><code>
bash infer.sh
</code></pre>

<h3>Git-LFS Setup</h3>
<pre><code>
git lfs install
git lfs track "*.pth"
</code></pre>

<hr/>

<h2>📄 License</h2>
<p>For academic and research use only.</p>

<br/>

<p align="center">
  <strong>© Image-Enhancer • ESRT Super-Resolution • For research & education</strong>
</p>
