import argparse
import os
import cv2
import numpy as np
from tqdm import tqdm
import torch

# Import project modules
from deepvision.models.adaptive_interpolator import AdaptiveInterpolator
from deepvision.models.rife_wrapper import RIFEInterpolator
from deepvision.models.optical_flow_wrapper import OpticalFlowInterpolator
from deepvision.pipelines.interpolate import bgr_uint8_to_t, t_to_bgr_uint8
from deepvision.metrics.psnr import psnr
from deepvision.metrics.ssim import ssim

try:
    from deepvision.metrics.lpips_net import lpips_distance
    _HAS_LPIPS = True
except:
    _HAS_LPIPS = False

def evaluate_split(manifest_path, data_root, interpolator, device='cuda'):
    """
    Reads the SNU-FILM manifest (txt file) and evaluates the model.
    """
    with open(manifest_path, 'r') as f:
        lines = f.readlines()

    psnr_list = []
    ssim_list = []
    lpips_list = []

    print(f"Evaluating on {manifest_path} ({len(lines)} triplets)...")

    for line in tqdm(lines):
        paths = line.strip().split()
        if len(paths) != 3:
            continue
        
        # Construct full image paths
        # Assumes data_root is the base directory containing 'data/SNU-FILM'
        p0 = os.path.join(data_root, paths[0])   # Input Frame 0
        pgt = os.path.join(data_root, paths[1])  # Ground Truth (Middle Frame)
        p1 = os.path.join(data_root, paths[2])   # Input Frame 1

        # Load images
        img0 = cv2.imread(p0)
        gt = cv2.imread(pgt)
        img1 = cv2.imread(p1)

        if img0 is None or gt is None or img1 is None:
            print(f"Warning: Image not found: {paths[0]}")
            continue

        # Convert to PyTorch tensors
        t0 = bgr_uint8_to_t(img0)
        t1 = bgr_uint8_to_t(img1)

        # Run Inference
        # The Adaptive model automatically decides whether to use RIFE or Optical Flow
        pred_tensor = interpolator(t0, t1, t=torch.tensor([0.5])) 
        
        pred_img = t_to_bgr_uint8(pred_tensor)

        # Calculate metrics
        psnr_list.append(psnr(pred_img, gt))
        ssim_list.append(ssim(pred_img, gt))
        
        if _HAS_LPIPS:
            lpips_list.append(lpips_distance(pred_img, gt, device=device))

    print("\n" + "="*30)
    print(f"Results for: {os.path.basename(manifest_path)}")
    print(f"Average PSNR: {np.mean(psnr_list):.4f} dB")
    print(f"Average SSIM: {np.mean(ssim_list):.4f}")
    if _HAS_LPIPS:
        print(f"Average LPIPS: {np.mean(lpips_list):.4f}")
    print("="*30 + "\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True, help='Path to test-extreme.txt or others')
    parser.add_argument('--data-root', required=True, help='Root folder where "data/SNU-FILM/..." exists')
    parser.add_argument('--mode', default='adaptive', choices=['adaptive', 'rife', 'flow'])
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Initialize the model
    if args.mode == 'adaptive':
        print("Loading Adaptive Model...")
        # Verify the import path for your specific RIFE implementation below
        interpolator = AdaptiveInterpolator(rife_impl="train_log.RIFE4_25.RIFE_HDv3:Model") 
    elif args.mode == 'rife':
        print("Loading RIFE Model...")
        interpolator = RIFEInterpolator().load_pretrained(impl="train_log.RIFE4_25.RIFE_HDv3:Model")
    elif args.mode == 'flow':
        print("Loading Optical Flow...")
        interpolator = OpticalFlowInterpolator()

    evaluate_split(args.manifest, args.data_root, interpolator, device)

if __name__ == "__main__":
    main()