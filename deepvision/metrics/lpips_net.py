"""LPIPS metric wrapper (optional). Install `lpips` to use this file."""
import torch
import lpips
import numpy as np
import cv2

_lpips = None

def _get_lpips(device: str = "cuda"):
    global _lpips
    if _lpips is None:
        _lpips = lpips.LPIPS(net='alex').to(device)
    return _lpips

def lpips_distance(img1, img2, device: str = "cuda") -> float:
    # Expect BGR uint8 in [0,255]
    net = _get_lpips(device)
    t1 = torch.from_numpy(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255.0
    t2 = torch.from_numpy(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255.0
    t1 = (t1 * 2 - 1).unsqueeze(0).to(device)
    t2 = (t2 * 2 - 1).unsqueeze(0).to(device)
    with torch.no_grad():
        d = net(t1, t2).mean().item()
    return float(d)
