"""Simple PSNR implementation for uint8 images."""
import numpy as np

def psnr(img1, img2, max_val: float = 255.0) -> float:
    diff = img1.astype(np.float32) - img2.astype(np.float32)
    mse = float((diff ** 2).mean()) + 1e-8
    return 20.0 * np.log10(max_val) - 10.0 * np.log10(mse)
