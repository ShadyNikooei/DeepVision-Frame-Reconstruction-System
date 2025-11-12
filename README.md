# DeepVision — Frame Reconstruction & Interpolation (Classical + RIFE)

This repository contains the source code for the **DeepVision Frame Reconstruction System**, a video frame reconstruction pipeline that now supports **two interchangeable paths**:
- a **classical computer vision (CV)** path (HSV-based keyframe detection + optical-flow-based interpolation), and  
- a **deep-learning path** using **RIFE v4.25** for high-quality frame interpolation.

> The original project focused on classical CV with HSV analysis, Farnebäck optical flow, and a hybrid interpolation stage. Those capabilities are preserved and remain selectable.


## Recent Changes (Nov 2025)

- **Integrated RIFE 4.25** under `train_log/RIFE4_25/` (model, flownet, warplayer, loss, ssim, vgg, util).
- **Device-safe execution:** hardcoded `.cuda()` calls replaced with `.to(device)` to support both **CPU** and **CUDA** runtimes.
- **Import hygiene:** absolute imports converted to **package-relative** (e.g., `.flownet`, `.warplayer`) so the model is importable as a package.
- **Optional profiling dep:** `torchstat` made **optional** (inference no longer fails if it’s missing).
- **Environment guidance:** documented a stable **Python 3.12 + CUDA 12.1** route (Windows, NVIDIA) plus a **CPU fallback**.
- **CLI alignment:** unified runtime flags with Practical-RIFE style (`--video/--img`, `--output`, `--model`, `--multi`, `--exp`, `--scale`, `--fps`, `--ext`, `--montage`).
- **Docs cleanup:** removed “will use deep learning in the future” notes—**DL interpolation is now implemented and enabled** as an alternative path.
- **Git hygiene:** added ignore patterns for venvs, site-packages, large artifacts (weights, videos, temporary folders).


## Project Overview

The pipeline is organized into three major components (classical path preserved; DL path added for step 3):

1. **Keyframe Extraction** — detects salient frames using HSV differences with a weighted moving average, plus a statistical threshold to mark keyframes.  
2. **Incomplete Video Generation** — simulates missing frames by dropping every other frame (models transmission loss / low frame rate).  
3. **Frame Reconstruction via Interpolation**  
   - **Classical**: Farnebäck optical flow + hybrid interpolation (fast `cv2.remap` + robust correction).  
   - **Deep Learning (new)**: **RIFE v4.25** produces perceptually high-quality intermediate frames.


## B.Sc. Project Summary (Classical CV Version — concise)

A brief, self-contained recap of the original Bachelor's project **without deep learning**, kept here for context and reproducibility.

### Components
- **Keyframe Extraction:** HSV-based differences → smoothed by a weighted moving average → statistical thresholding picks keyframes.  
- **Incomplete Video Generation:** drops every other frame to emulate packet loss / low-FPS capture.  
- **Frame Reconstruction (Classical):** Farnebäck optical flow + **hybrid interpolation** (fast `cv2.remap` warping + SciPy `RegularGridInterpolator` for edge/boundary corrections).

### Current Methodology (Classical)
- HSV analysis for frame similarity
- Optical flow for motion estimation
- Hybrid interpolation for accuracy & speed  
_No neural models in this version; architecture was designed to allow future DL integration._

### Output Backends
- **OpenCV-based** (fast `.mp4`), **ImageIO-based** (flexible: `.gif`, `.webm`, …). Choose inside code via a flag.

### Typical Folder Structure (classical-only minimal)
```
project/
├── main.py
├── keyframe_extraction.py
├── make_incomplete_video.py
├── hybrid_optical_flow_interpolation.py
├── reconstruct_full_video.py
└── output/
```

### How to Run (classical pipeline)
```bash
python main.py
# (ensure input/output paths in main.py are set)
```

### Requirements (classical)
Python ≥3.7, OpenCV, NumPy, SciPy, ImageIO.  
Install with:
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


## Folder Structure (full, with RIFE)

```
project/
│
├── main.py                               # entry point for the end-to-end pipeline
├── keyframe_extraction.py                # HSV-based keyframe detection (classical)
├── make_incomplete_video.py              # drops every other frame
├── hybrid_optical_flow_interpolation.py  # classical hybrid interpolation
├── reconstruct_full_video.py             # assembles the final video
├── inference_video.py                    # RIFE inference (video or PNG sequence)
├── train_log/
│   └── RIFE4_25/                         # RIFE v4.25 package (model, flownet, warplayer, loss, ssim, vgg, util, …)
└── output/ | vid_out/ | temp/            # results and temporary assets
```

> The classical layout above follows the original project organization; RIFE was added as a drop-in interpolator without removing the CV path.


## Setup

### Option A — GPU (Windows, NVIDIA)
1) Create a Python **3.12** virtual environment and upgrade `pip`.  
2) Install **PyTorch CUDA 12.1** wheels (torch/vision/audio for cp312, win_amd64).  
3) Install core deps: `opencv-python`, `numpy`, `tqdm`, `scikit-video`, `imageio`, `imageio-ffmpeg`, `moviepy`, `pytorch-msssim`.  
4) (Optional) `pip install torchstat`.

> If `torch.cuda.is_available()` returns `False`, update your NVIDIA driver (CUDA 12.1 compatible).

### Option B — CPU (fallback)
Create a venv, install CPU wheels of torch/vision/audio from PyPI, then the same core deps above.


## Usage (RIFE examples)

### Interpolate a video with RIFE (2×)
```powershell
python .\inference_video.py --video .\input.mp4 --multi 2 --model .\train_log\RIFE4_25
# -> input_2X_<fps>.mp4 (same folder unless --output is provided)
```

### Higher FPS without merging audio (lighter I/O)
```powershell
python .\inference_video.py --video .\input.mp4 --multi 2 --fps 60 --model .\train_log\RIFE4_25
```

### Low-VRAM / laptop GPUs (reduce memory & heat)
```powershell
python .\inference_video.py --video .\input.mp4 --multi 2 --scale 0.5 --model .\train_log\RIFE4_25
```

### PNG sequence input
```powershell
python .\inference_video.py --img .\frames\ --multi 4 --model .\train_log\RIFE4_25
# frames/0.png ... N.png (numeric ordering)
```

**Flag reference**

* `--multi K` → insert K−1 frames between each pair (`K=2` → 2×).
* `--exp E` → interpolation factor = `2^E` (e.g., `E=2` → 4×).
* `--scale S` → internal processing scale (use `0.5` for 4K or low-VRAM).
* `--fps F` → set output FPS (disables audio merge if provided).
* `--model PATH` → points to `train_log/RIFE4_25`.


## Output Backends

* **OpenCV-based** (fast, typical `.mp4`)
* **ImageIO-based** (flexible formats like `.gif`, `.webm`)

You can choose the backend inside the code depending on platform/output needs. 


## Evaluation (report-ready)

Compute:

* **PSNR / SSIM** (fidelity) on a held-out clip; average across frames.
* **LPIPS** (perceptual similarity; lower is better).
* **Temporal consistency** (e.g., SSIM/LPIPS across time to reveal flicker).
* **Runtime & memory** (FPS, peak VRAM) for `{scale ∈ {1.0, 0.5, 0.25}} × {multi ∈ {2,3,4}}`, CPU vs CUDA.

Suggested table columns: *Resolution, Scale, Factor, Device, PSNR, SSIM, LPIPS, FPS, Peak VRAM*.


## Troubleshooting

* **`Torch not compiled with CUDA`** → you installed CPU wheels; install cu121 wheels or run CPU intentionally.
* **`ModuleNotFoundError: torchstat`** → optional; install `torchstat` or keep profiling disabled.
* **Laptop restarts under load** → reduce `--scale` (0.5/0.25), process short clips, cool the device, update NVIDIA driver.
* **PowerShell uses Store Python** → disable App execution aliases or call venv interpreter directly.


## Git Hygiene

Add to `.gitignore` (recommended):

```
__pycache__/
*.pyc
.venv*/              # all virtual envs
**/Lib/site-packages/
*.pth *.pkl *.pt     # model weights
*.mp4                # video outputs
vid_out/ temp/       # generated/temporary folders
```

If you previously embedded a nested repo (e.g., `Practical-RIFE`), either **vendor** it (remove inner `.git`) or add as a **submodule**.


## Author

**Shady Nikooei** — Final Year B.Sc. Student in Computer Engineering.  
The project now offers both **classical CV** and **RIFE-based deep interpolation** in a single, unified pipeline.
