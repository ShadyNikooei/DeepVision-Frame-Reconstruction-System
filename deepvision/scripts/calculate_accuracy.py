import argparse
import cv2
import numpy as np
import torch
from deepvision.metrics.psnr import psnr
from deepvision.metrics.ssim import ssim

# Attempt to import LPIPS module (available in your project files)
try:
    from deepvision.metrics.lpips_net import lpips_distance
    _HAS_LPIPS = True       
except ImportError:
    _HAS_LPIPS = False
    print("[WARNING] 'lpips' library not found. Skipping LPIPS metric.")

def read_frames(video_path):
    """Reads all frames from a video file and returns them as a list."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def main():
    parser = argparse.ArgumentParser(description="Calculate accuracy metrics (PSNR/SSIM/LPIPS) between two videos.")
    parser.add_argument('--original', required=True, help="Path to the original ground-truth video.")
    parser.add_argument('--reconstructed', required=True, help="Path to the reconstructed output video.")
    args = parser.parse_args()

    # Determine processing device (GPU is recommended for LPIPS)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[INFO] Processing on: {device.upper()}")

    print(f"[INFO] Reading original video: {args.original}")
    orig_frames = read_frames(args.original)
    
    print(f"[INFO] Reading reconstructed video: {args.reconstructed}")
    recon_frames = read_frames(args.reconstructed)

    min_len = min(len(orig_frames), len(recon_frames))
    if len(orig_frames) != len(recon_frames):
        print(f"[WARNING] Frame count mismatch! Orig: {len(orig_frames)}, Recon: {len(recon_frames)}")
        print(f"[INFO] Evaluating only the first {min_len} frames...")

    psnr_scores = []
    ssim_scores = []
    lpips_scores = []

    print("[INFO] Computing metrics... (This may take a while for LPIPS)")
    
    for i in range(min_len):
        # 1. Calculate PSNR
        p = psnr(orig_frames[i], recon_frames[i])
        psnr_scores.append(p)
        
        # 2. Calculate SSIM
        s = ssim(orig_frames[i], recon_frames[i])
        ssim_scores.append(s)
        
        # 3. Calculate LPIPS (if available)
        if _HAS_LPIPS:
            # lpips_distance function is defined in lpips_net.py
            l = lpips_distance(orig_frames[i], recon_frames[i], device=device)
            lpips_scores.append(l)

        # Show progress every 10 frames
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{min_len} frames...", end='\r')

    avg_psnr = np.mean(psnr_scores)
    avg_ssim = np.mean(ssim_scores)
    
    print("\n" + "="*50)
    print("          ACCURACY REPORT          ")
    print("="*50)
    print(f" Average PSNR:  {avg_psnr:.2f} dB  (Higher is Better)")
    print(f" Average SSIM:  {avg_ssim:.4f}     (Higher is Better, Max=1.0)")
    
    if _HAS_LPIPS:
        avg_lpips = np.mean(lpips_scores)
        print(f" Average LPIPS: {avg_lpips:.4f}     (Lower is Better)")
    else:
        print(" Average LPIPS: N/A (Install 'lpips' to enable)")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()