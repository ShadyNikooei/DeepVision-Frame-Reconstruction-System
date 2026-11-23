"""A pragmatic SSIM approximation using OpenCV; good enough for comparisons."""
import numpy as np
import cv2

def ssim(img1, img2) -> float:
    # Convert to luma (Y) plane
    y1 = cv2.cvtColor(img1, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float32)
    y2 = cv2.cvtColor(img2, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float32)
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu1 = cv2.GaussianBlur(y1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(y2, (11, 11), 1.5)
    mu1_sq, mu2_sq, mu12 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    sigma1_sq = cv2.GaussianBlur(y1 * y1, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(y2 * y2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(y1 * y1, (11, 11), 1.5) - mu12
    ssim_map = ((2 * mu12 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return float(np.mean(ssim_map))
