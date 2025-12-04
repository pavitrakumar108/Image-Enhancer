<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Image-Enhancer — ESRT Super-Resolution</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    :root{
      --bg:#0b0d10; --card:#111418; --muted:#9aa4b2; --fg:#e8ecf1;
      --brand:#7cd1ff; --accent:#7ef7c2; --codebg:#0f1317; --border:#1c232b;
    }
    *{box-sizing:border-box}
    html,body{margin:0;padding:0;background:var(--bg);color:var(--fg);font:16px/1.6 system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial}
    a{color:var(--brand);text-decoration:none}
    a:hover{text-decoration:underline}
    .wrap{max-width:980px;margin:40px auto;padding:0 20px}
    header{display:flex;align-items:center;gap:14px;margin-bottom:24px}
    .logo{width:40px;height:40px;border-radius:10px;background:linear-gradient(135deg,var(--brand),var(--accent));box-shadow:0 8px 30px rgba(124,209,255,.35)}
    h1{margin:.2rem 0 0;font-size:2rem}
    .tag{display:inline-block;margin-top:6px;padding:4px 10px;border:1px solid var(--border);border-radius:999px;color:var(--muted);font-size:.9rem}
    .cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:16px;margin:24px 0}
    .card{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:16px}
    .card h3{margin:.2rem 0 8px;font-size:1.05rem}
    .muted{color:var(--muted)}
    h2{margin:28px 0 12px;font-size:1.4rem}
    .section{background:var(--card);border:1px solid var(--border);border-radius:16px;padding:18px;margin:18px 0}
    .klist{margin:0;padding-left:1.1rem}
    .klist li{margin:.35rem 0}
    pre{background:var(--codebg);border:1px solid var(--border);border-radius:12px;padding:14px;overflow:auto}
    code{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,Monaco,monospace}
    .grid-2{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px}
    .cta{display:inline-block;margin-top:8px;padding:10px 14px;border-radius:10px;background:linear-gradient(135deg,var(--brand),var(--accent));color:#0a0e12;font-weight:600}
    .pill{display:inline-block;padding:4px 8px;border-radius:999px;background:#0e1318;border:1px solid var(--border);color:var(--muted);font-size:.85rem;margin-right:6px}
    hr{border:none;border-top:1px solid var(--border);margin:24px 0}
    footer{margin:24px 0;color:var(--muted);font-size:.9rem}
  </style>
</head>
<body>
  <div class="wrap">
    <header>
      <div class="logo" aria-hidden="true"></div>
      <div>
        <h1>Image-Enhancer</h1>
        <div class="tag">Super-Resolution with ESRT (Efficient Super-Resolution Transformer)</div>
      </div>
    </header>

    <div class="section">
      <p><strong>Image-Enhancer</strong> is a clean, modular implementation of the ESRT architecture for generating high-resolution, sharp images from low-resolution inputs. The repo includes training, evaluation (PSNR/SSIM), inference scripts, and dataset utilities with a research-friendly structure.</p>
      <div class="cards">
        <div class="card">
          <h3>Highlights</h3>
          <ul class="klist">
            <li>Full ESRT pipeline (train / eval / infer)</li>
            <li>PSNR &amp; SSIM evaluation</li>
            <li>Git-LFS for large checkpoints</li>
            <li>Dataset &amp; visualization tools</li>
          </ul>
        </div>
        <div class="card">
          <h3>Why ESRT?</h3>
          <p class="muted">Combines lightweight Transformers + CNNs with kernel-aware modeling for speed, low memory, and strong detail preservation.</p>
        </div>
      </div>
    </div>

    <h2>Overview</h2>
    <div class="section">
      <p>Traditional SR struggles with real-world noise, blur, and edge loss. ESRT addresses these using:</p>
      <ul class="klist">
        <li>Lightweight Transformer blocks</li>
        <li>CNN residual feature extractors</li>
        <li>Hierarchical feature fusion</li>
        <li>Kernel-aware processing</li>
      </ul>
    </div>

    <h2>Dataset</h2>
    <div class="section grid-2">
      <div>
        <p>Uses <strong>DF2K</strong> (DIV2K + Flickr2K + OST). Place locally:</p>
        <pre><code>datasets/DF2K/</code></pre>
        <p class="muted">This folder is ignored by Git (.gitignore). Download datasets manually.</p>
      </div>
      <div>
        <span class="pill">DIV2K</span>
        <span class="pill">Flickr2K</span>
        <span class="pill">OST</span>
      </div>
    </div>

    <h2>File Structure</h2>
    <div class="section">
<pre><code>Image-Enhancer/
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
└── README.md                           # Global project documentation
</code></pre>
    </div>

    <h2>Quick Start</h2>
    <div class="section grid-2">
      <div>
        <h3>Train</h3>
        <pre><code>cd models/esrt/scripts
bash train.sh</code></pre>
      </div>
      <div>
        <h3>Evaluate (PSNR / SSIM)</h3>
        <pre><code>bash validate.sh</code></pre>
      </div>
      <div>
        <h3>Inference</h3>
        <pre><code>bash infer.sh</code></pre>
      </div>
      <div>
        <h3>Git-LFS</h3>
        <pre><code>git lfs install
git lfs track "*.pth"</code></pre>
      </div>
    </div>

    <hr />
    <footer>
      <div>© Image-Enhancer • ESRT Super-Resolution • For research &amp; education.</div>
    </footer>
  </div>
</body>
</html>
