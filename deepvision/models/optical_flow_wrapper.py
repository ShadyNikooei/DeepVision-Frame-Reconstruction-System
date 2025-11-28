
import torch
import cv2
import numpy as np
import sys
import os

# --- PATH FIX ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
cv_folder_candidates = [
    os.path.join(project_root, 'computer_vision'),
    os.path.join(project_root, 'computer_vision_')
]

for folder in cv_folder_candidates:
    if os.path.exists(folder) and folder not in sys.path:
        sys.path.append(folder)
# ----------------

try:
    from hybrid_optical_flow_interpolation import hybrid_optical_flow_interpolation
except ImportError:
    try:
        from computer_vision.hybrid_optical_flow_interpolation import hybrid_optical_flow_interpolation
    except ImportError:
        from computer_vision_.hybrid_optical_flow_interpolation import hybrid_optical_flow_interpolation

from deepvision.pipelines.interpolate import t_to_bgr_uint8, bgr_uint8_to_t

class OpticalFlowInterpolator(torch.nn.Module):
    def __init__(self, fix_border: bool = True):
        super().__init__()
        self.fix_border = fix_border
        self.device = "cpu"

    def forward(self, img0: torch.Tensor, img1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        img0_bgr_np = t_to_bgr_uint8(img0)
        img1_bgr_np = t_to_bgr_uint8(img1)
        frame_a = cv2.cvtColor(img0_bgr_np, cv2.COLOR_BGR2RGB)
        frame_c = cv2.cvtColor(img1_bgr_np, cv2.COLOR_BGR2RGB)
        interpolated_rgb_np = hybrid_optical_flow_interpolation(frame_a, frame_c, self.fix_border)
        return bgr_uint8_to_t(cv2.cvtColor(interpolated_rgb_np, cv2.COLOR_RGB2BGR))

    def load_pretrained(self, **kwargs) -> "OpticalFlowInterpolator":
        return self
