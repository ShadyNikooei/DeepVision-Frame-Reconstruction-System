import argparse
import os
import numpy as np
import cv2
from typing import List
# Reuse existing I/O and Metrics utilities
from deepvision.utils.io import read_video_frames
from deepvision.metrics.psnr import psnr
from deepvision.metrics.ssim import ssim

try:
    from deepvision.metrics.lpips_net import lpips_distance
    _HAS_LPIPS = True
except Exception:
    _HAS_LPIPS = False

def calculate_metrics(ref_frames: List[np.ndarray], test_frames: List[np.ndarray], name: str):
    """Calculates PSNR, SSIM, and LPIPS for two video frame lists."""
    if len(ref_frames) != len(test_frames):
        print(f"[{name}] ERROR: Frame count mismatch ({len(ref_frames)} vs {len(test_frames)})")
        return 

    psnrs, ssims, lpipss = [], [], []
    
    for ref, test in zip(ref_frames, test_frames):
        # Calculate metrics
        psnrs.append(psnr(ref, test))
        ssims.append(ssim(ref, test))
        
        if _HAS_LPIPS:
            # LPIPS requires BGR and is best run on GPU
            dev = 'cuda' if cv2.cuda.getCudaEnabledDeviceCount() > 0 else 'cpu'
            lpipss.append(lpips_distance(ref, test, device=dev))

    def avg(x):
        return float(np.mean(x)) if x else float('nan')

    print(f"\n--- Results for {name} ---")
    print(f"Total Frames: {len(ref_frames)}")
    print(f"Average PSNR: {avg(psnrs):.3f} dB")
    print(f"Average SSIM: {avg(ssims):.4f}")
    if _HAS_LPIPS:
        print(f"Average LPIPS: {avg(lpipss):.4f} (Lower is better)")
    else:
        print("LPIPS: (Skipped — install `lpips` library)")

def main():
    p = argparse.ArgumentParser(description="Compares reconstructed videos against a full reference video.")
    p.add_argument('--reference', required=True, help='Path to the original (complete) reference video.')
    p.add_argument('--test-videos', nargs='+', required=True, help='List of paths to reconstructed videos for comparison.')
    p.add_argument('--names', nargs='+', required=True, help='Names of the methods in order (e.g., Flow, RIFE, Adaptive).')
    args = p.parse_args()

    if len(args.test_videos) != len(args.names):
        raise ValueError("The number of test videos and their names must be equal.")

    print("Reading reference video frames...")
    ref_frames = read_video_frames(args.reference)
    
    for video_path, name in zip(args.test_videos, args.names):
        print(f"Reading test video frames: {name}...")
        test_frames = read_video_frames(video_path)
        
        # Compare each video against the reference
        calculate_metrics(ref_frames, test_frames, name)

if __name__ == '__main__':
    main()