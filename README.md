# DeepVision — Frame Reconstruction & Interpolation (Classical + RIFE + Adaptive)

DeepVision is a **video frame reconstruction pipeline** that supports **three interchangeable interpolation paths** in a single, unified system:

- **Classical CV path**  
  HSV-based keyframe detection + Farnebäck optical-flow interpolation.
- **Deep-learning path (RIFE v4.25)**  
  High-quality frame interpolation using RIFE as a benchmark model.
- **Adaptive Switching path (novel)**  
  Intelligent runtime selection between Classical Flow and RIFE based on motion, optimizing **speed vs. quality**.

The original project was a **classical computer vision B.Sc. project**. All classical capabilities are still available and now wrapped to integrate cleanly with the deep-learning and adaptive pipelines.

---

## Table of Contents

1. [Recent Changes (Nov 2025)](#recent-changes-nov-2025)  
2. [Project Overview](#project-overview)  
3. [B.Sc. Project Summary (Classical CV)](#bsc-project-summary-classical-cv)  
4. [Folder Structure](#folder-structure)  
5. [Setup](#setup)  
6. [Usage & Evaluation](#usage-and-evaluation)  
7. [Output Backends](#output-backends)  
8. [Evaluation Protocol (Report-Ready)](#evaluation-report-ready)  
9. [Troubleshooting](#troubleshooting)  
10. [Git Hygiene](#git-hygiene)  
11. [Author](#author)  


## Recent Changes (Nov 2025)

- **Adaptive Switching System**
  - Introduced `AdaptiveInterpolator` for dynamic selection between Classical Flow and RIFE based on estimated motion.
- **Classical Wrapper Integration**
  - Added `OpticalFlowInterpolator` (in `optical_flow_wrapper.py`) to integrate the original CV method into the new pipeline.
- **Evaluation CLI**
  - Added `deepvision/scripts/eval_interpolator.py` for quantitative metrics (**PSNR, SSIM, LPIPS**) across all methods.
- **RIFE 4.25 Integration**
  - Integrated RIFE 4.25 under `train_log/RIFE4_25/`.
- **Device-Safe Execution**
  - Replaced hardcoded `.cuda()` calls with `.to(device)` to support **CPU** and **CUDA**.
- **Import Hygiene**
  - Converted absolute imports to **package-relative** (e.g., `.flownet`, `.warplayer`) so the model is importable as a package.
- **Optional Profiling Dependency**
  - `torchstat` is now **optional** (inference will not fail if it’s missing).
- **Environment Guidance**
  - Documented a stable **Python 3.12 + CUDA 12.1** setup (Windows + NVIDIA) and a **CPU fallback**.
- **CLI Alignment**
  - Unified runtime flags with Practical-RIFE style:  
    `--video/--img`, `--output`, `--model`, `--multi`, `--exp`, `--scale`, `--fps`, `--ext`, `--montage`.
- **Docs Cleanup**
  - Removed “will use deep learning in the future” wording — **DL interpolation is now implemented and enabled**.
- **Git Hygiene**
  - Added ignore patterns for venvs, site-packages, large artifacts (weights, videos, temp folders).


## Project Overview

The full pipeline is organized into three major components:

1. **Keyframe Extraction**  
   Detects salient frames using HSV differences + weighted moving average + statistical thresholding.

2. **Incomplete Video Generation**  
   Simulates missing frames by dropping every other frame (models transmission loss / low FPS).

3. **Frame Reconstruction via Interpolation**
   - **Classical Wrapper (Updated)**  
     Original Farnebäck optical flow + hybrid interpolation, now wrapped in  
     `deepvision/models/optical_flow_wrapper.py` for direct integration and comparison.
   - **Deep Learning (RIFE v4.25)**  
     Serves as the **high-accuracy benchmark** for frame interpolation.
   - **Adaptive Switching System (New)**  
     An intelligent system that **dynamically switches** between:
       - fast Classical Wrapper (low motion), and  
       - accurate RIFE model (high motion)  
     to optimize both **runtime and quality**.


## B.Sc. Project Summary (Classical CV)

This is a brief, self-contained recap of the original **Bachelor’s** project (without deep learning), kept for context and reproducibility.

### Components

- **Keyframe Extraction**
  - HSV-based frame differences  
  - Smoothed via weighted moving average  
  - Statistical thresholding to select keyframes
- **Incomplete Video Generation**
  - Drops every other frame to emulate packet loss / low-FPS capture.
- **Frame Reconstruction (Classical)**
  - Farnebäck optical flow  
  - **Hybrid interpolation**:
    - fast `cv2.remap` warping
    - SciPy `RegularGridInterpolator` for edge/boundary corrections

### Classical Methodology

- HSV analysis for frame similarity  
- Optical flow for motion estimation  
- Hybrid interpolation for **accuracy & speed**  

> **Note:** The original architecture explicitly anticipated future deep-learning integration. No neural models were used in the initial B.Sc. version.

### Classical Output Backends

- **OpenCV-based**: fast `.mp4` output  
- **ImageIO-based**: flexible formats (`.gif`, `.webm`, …), configurable via a flag in code.

### Minimal Classical-Only Folder Structure

```markdown
project/
├── main.py
├── keyframe_extraction.py
├── make_incomplete_video.py
├── hybrid_optical_flow_interpolation.py
├── reconstruct_full_video.py
└── output/
```

### How to Run (Classical-Only Pipeline)

```bash
python main.py
# (ensure input/output paths inside main.py are set correctly)
```

### Classical Requirements

- Python ≥ 3.7  
- OpenCV  
- NumPy  
- SciPy  
- ImageIO  

Install via:

```bash
pip install -r requirements.txt
```

### Applications & AI-Oriented Dataset Prep
- Recover damaged/incomplete streams; frame-rate upscaling (e.g., 30→60 fps).
- Summarization via keyframes; temporally consistent preprocessing for ML.
- Great for preparing datasets for **action recognition, gesture detection, temporal segmentation, video captioning**.

### Future Plans (from original scope)
- Add DL-based interpolation (e.g., **RIFE**, **Super SloMo**), real-time restoration.
- Evaluate on challenging real-world datasets.

---

## Folder Structure (full, with RIFE)

```
project/
│
├── computer_vision_/                      # Top-level classical/utility scripts
│   ├── hybrid_optical_flow_interp...py
│   ├── keyframe_extraction.py
│   ├── main.py
│   └── make_incomplete_video.py
├── deepvision/
│   ├── metrics/
│   │   ├── lpips_net.py
│   │   ├── psnr.py
│   │   └── ssim.py
│   ├── models/
│   │   ├── adaptive_interpolator.py    # Core logic for intelligent Flow/RIFE switching.
│   │   ├── optical_flow_wrapper.py     # Wrapper for the classical CV interpolation method.
│   │   └── rife_wrapper.py
│   ├── pipelines/
│   │   └── interpolate.py
│   └── scripts/
│       ├── eval_interpolator.py        # CLI for PSNR/SSIM/LPIPS vs. Ground Truth.
│       └── infer_video.py              # Main CLI for interpolation (Flow, RIFE, Adaptive).
├── train_log/
│   └── RIFE4_25/                       # RIFE v4.25 package (model, flownet, warplayer, loss, etc.)
└── output/ | vid_out/ | temp/          # Results and temporary assets
```


## Setup

### Option A — GPU (Windows, NVIDIA)

1. Create a **Python 3.12** virtual environment and upgrade `pip`.
2. Install **PyTorch CUDA 12.1** wheels (torch / torchvision / torchaudio) for:
   - `cp312`
   - `win_amd64`
3. Install core dependencies:
   - `opencv-python`
   - `numpy`
   - `tqdm`
   - `scikit-video`
   - `imageio`
   - `imageio-ffmpeg`
   - `moviepy`
   - `pytorch-msssim`
   - **`lpips`**
4. (Optional) Install profiling dependency:
   ```bash
   pip install torchstat
   ```

> If `torch.cuda.is_available()` returns `False`, update your NVIDIA driver to a **CUDA 12.1–compatible** version.

### Option B — CPU (Fallback)

1. Create a virtual environment (any supported OS).  
2. Install **CPU-only** wheels of `torch`, `torchvision`, `torchaudio` from PyPI.  
3. Install the same core dependencies as above, including `lpips`.


## Usage and Evaluation

All operations assume an **incomplete input video** (frames dropped or lost), typically generated via `make_incomplete_video.py`.

### 1. Incomplete Video Generation

Create a deficient video (e.g., periodic loss):

```bash
python computer_vision_/make_incomplete_video.py     --input input_video.mp4     --output incomplete_video.mp4     --loss-pattern periodic     --loss-ratio 0.5
```

---

### 2. Execution Modes (Using `infer_video.py`)

The main script:

```bash
python deepvision/scripts/infer_video.py [MODE FLAGS] ...
```

#### A. Classic Optical Flow (Benchmark)

Runs the classical method via the new wrapper:

```bash
python deepvision/scripts/infer_video.py     --input incomplete_video.mp4     --output flow_out.mp4     --use-flow
```

#### B. RIFE Deep Learning (Accuracy Benchmark)

Runs the RIFE v4.25 model. Requires the implementation module and checkpoint:

```bash
python deepvision/scripts/infer_video.py     --input incomplete_video.mp4     --output rife_out.mp4     --impl [RIFE_MODULE]     --ckpt [CHECKPOINT_PATH]
```

#### C. Adaptive Switching System (Innovation)

Runs the intelligent system that chooses Flow vs. RIFE per region/scene based on motion estimation:

```bash
python deepvision/scripts/infer_video.py     --input incomplete_video.mp4     --output adaptive_out.mp4     --adaptive     --impl [RIFE_MODULE]     --ckpt [CHECKPOINT_PATH]
```

---

### 3. Quantitative Evaluation

After generating reconstructed videos, compute **PSNR**, **SSIM**, and **LPIPS** using:

```bash
python deepvision/scripts/eval_interpolator.py     --reference original_video.mp4     --test-videos flow_out.mp4 rife_out.mp4 adaptive_out.mp4     --names Flow RIFE Adaptive
```

This script compares each test video against the **ground-truth original** and reports metrics suitable for inclusion in your final report.


## Output Backends

You can configure which backend to use inside the code:

- **OpenCV-based**
  - Fast `.mp4` export  
  - Good default for most platforms
- **ImageIO-based**
  - Flexible export formats: `.gif`, `.webm`, etc.  
  - Useful for web demos or visualizations


## Evaluation (Report-Ready)

For a thorough evaluation, compute:

- **PSNR / SSIM**  
  - On a held-out clip  
  - Average across frames  
- **LPIPS**  
  - Perceptual similarity (lower is better)  
- **Temporal Consistency**
  - SSIM/LPIPS across time to expose flicker or jitter  
- **Runtime & Memory**
  - Frames per second (FPS)  
  - Peak VRAM usage  

Recommended experimental grid:

- **Scale**: `{1.0, 0.5, 0.25}`  
- **Multi-factor (`--multi`)**: `{2, 3, 4}`  
- **Device**: CPU vs. CUDA

Suggested table columns for the report:

> `Resolution | Scale | Factor | Device | PSNR | SSIM | LPIPS | FPS | Peak VRAM`


## Troubleshooting

- **`Torch not compiled with CUDA`**
  - You likely installed CPU-only wheels.  
  - Install CUDA 12.1 wheels for PyTorch, or run explicitly in CPU mode.

- **`ModuleNotFoundError: torchstat`**
  - `torchstat` is optional.  
  - Either install it:
    ```bash
    pip install torchstat
    ```
    or keep profiling features disabled.

- **Laptop restarts / thermal throttling**
  - Reduce `--scale` to `0.5` or `0.25`.  
  - Process shorter clips.  
  - Improve cooling and update NVIDIA drivers.

- **PowerShell starts Microsoft Store Python**
  - Disable App Execution Aliases in Windows settings,  
  - or call the venv’s Python directly (e.g., `.\.venv\Scripts\python.exe`).


## Git Hygiene

Recommended `.gitignore` additions:

```gitignore
__pycache__/
*.pyc

.venv*/                 # all virtual envs
**/Lib/site-packages/

*.pth
*.pkl
*.pt                    # model weights

*.mp4                   # video outputs
vid_out/
temp/                   # generated/temporary folders
```

If you previously embedded a nested repo (e.g., `Practical-RIFE` inside this project):

- **Vendor** it: remove the inner `.git` folder  
  **or**
- Add it as a **git submodule** instead.


## Author

**Shady Nikooei**  
Final-Year B.Sc. Student in Computer Engineering  

This project now combines:

- **Classical Computer Vision**,  
- **RIFE-based Deep Interpolation**, and  
- a **novel Adaptive Switching System**  

into one cohesive, extensible frame reconstruction pipeline.
