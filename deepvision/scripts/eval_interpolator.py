"""
Evaluation script on triplet folders: A (000.png), B (001.png GT), C (002.png).
It compares the predicted middle frame (from A & C) against GT (B) using PSNR/SSIM
and optionally LPIPS if installed.
"""
import argparse, os, glob
import cv2
import numpy as np
from deepvision.models.rife_wrapper import RIFEInterpolator
from deepvision.pipelines.interpolate import bgr_uint8_to_t, t_to_bgr_uint8
from deepvision.metrics.psnr import psnr
from deepvision.metrics.ssim import ssim

try:
    from deepvision.metrics.lpips_net import lpips_distance
    _HAS_LPIPS = True
except Exception:
    _HAS_LPIPS = False

def read_triplet(folder: str):
    A = cv2.imread(os.path.join(folder, '000.png'))
    B = cv2.imread(os.path.join(folder, '001.png'))  # Ground truth middle frame
    C = cv2.imread(os.path.join(folder, '002.png'))
    if A is None or B is None or C is None:
        raise RuntimeError(f"Invalid triplet in {folder}")
    return A, B, C

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True, help='Path to directory containing triplet subfolders')
    p.add_argument('--impl', default=None, help='Real RIFE impl as module:Class (optional)')
    p.add_argument('--ckpt', default=None, help='Optional checkpoint path')
    args = p.parse_args()

    interpolator = RIFEInterpolator().load_pretrained(impl=args.impl, ckpt_path=args.ckpt)

    folders = sorted([d for d in glob.glob(os.path.join(args.root, '*')) if os.path.isdir(d)])
    psnrs, ssims, lpipss = [], [], []

    for fd in folders:
        A, B, C = read_triplet(fd)
        tA, tC = bgr_uint8_to_t(A), bgr_uint8_to_t(C)
        pred = interpolator.interpolate(tA, tC, t=0.5)
        P = t_to_bgr_uint8(pred)

        psnrs.append(psnr(P, B))
        ssims.append(ssim(P, B))
        if _HAS_LPIPS:
            dev = 'cuda' if cv2.cuda.getCudaEnabledDeviceCount() > 0 else 'cpu'
            lpipss.append(lpips_distance(P, B, device=dev))

    def avg(x):
        return float(np.mean(x)) if x else float('nan')

    print(f"Triplets: {len(folders)}")
    print(f"PSNR  : {avg(psnrs):.3f}")
    print(f"SSIM  : {avg(ssims):.4f}")
    if _HAS_LPIPS:
        print(f"LPIPS : {avg(lpipss):.4f} (lower is better)")
    else:
        print("LPIPS : (skipped — install `lpips` to enable)")

if __name__ == '__main__':
    main()
