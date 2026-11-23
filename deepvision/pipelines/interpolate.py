"""
Video interpolation pipeline glue code:
  * Converts OpenCV BGR frames <-> PyTorch tensors
  * Calls a provided interpolator (RIFE wrapper) to generate intermediates
"""
from typing import List
import numpy as np
import torch
import cv2

def bgr_uint8_to_t(img_bgr: np.ndarray) -> torch.Tensor:
    """(H,W,C) BGR uint8 -> (1,3,H,W) RGB float32 in [0,1]."""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    return t.unsqueeze(0)

def t_to_bgr_uint8(t: torch.Tensor) -> np.ndarray:
    """(1,3,H,W) RGB [0,1] -> (H,W,C) BGR uint8."""
    t = t.squeeze(0).clamp(0, 1)
    rgb = (t.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bgr

def interpolate_video_frames(frames: List[np.ndarray], interpolator, num_intermediate: int = 1) -> List[np.ndarray]:
    """Insert `num_intermediate` frames between each consecutive pair using `interpolator`.

    Args:
        frames: list of BGR uint8 frames
        interpolator: object with method `.interpolate(tA, tB, t)`
        num_intermediate: how many in-between frames to generate per pair
    Returns:
        New frame list with intermediates inserted.
    """
    if len(frames) < 2:
        return frames

    out: List[np.ndarray] = []
    for i in range(len(frames) - 1):
        a = frames[i]
        c = frames[i + 1]
        out.append(a)
        if num_intermediate > 0:
            tA = bgr_uint8_to_t(a)
            tC = bgr_uint8_to_t(c)
            for k in range(1, num_intermediate + 1):
                alpha = k / (num_intermediate + 1)
                pred = interpolator.interpolate(tA, tC, t=alpha)
                out.append(t_to_bgr_uint8(pred))
    out.append(frames[-1])
    return out
