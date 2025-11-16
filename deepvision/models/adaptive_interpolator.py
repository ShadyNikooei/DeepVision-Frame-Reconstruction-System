
import torch
import cv2
import numpy as np
from deepvision.models.rife_wrapper import RIFEInterpolator
from optical_flow_wrapper import OpticalFlowInterpolator
from deepvision.pipelines.interpolate import t_to_bgr_uint8, bgr_uint8_to_t

class AdaptiveInterpolator(torch.nn.Module):
    """
    Adaptive Interpolator: Switches between the classic Optical Flow method and 
    the RIFE Deep Learning model based on the estimated motion magnitude between frames.
    """
    def __init__(self, rife_impl=None, rife_ckpt=None, motion_threshold: float = 5.0):
        super().__init__()
        # Load both interpolation models
        self.rife = RIFEInterpolator().load_pretrained(impl=rife_impl, ckpt_path=rife_ckpt)
        self.flow_classic = OpticalFlowInterpolator()
        self.motion_threshold = motion_threshold
        # The device (CPU/CUDA) is determined by the RIFE model's setup
        self.device = self.rife.device
        print(f"Adaptive switch threshold set to: {self.motion_threshold} (Mean Flow Magnitude)")

    @torch.no_grad()
    def forward(self, img0: torch.Tensor, img1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Processes two input frames and returns an interpolated frame using the selected method.
        
        Args:
            img0, img1: Input frames as Tensors (1, C, H, W), RGB, float [0, 1].
            t: Time parameter for interpolation (e.g., 0.5 for the middle frame).
            
        Returns:
            interpolated: Output frame as a Tensor (1, C, H, W), RGB, float [0, 1].
        """
        # 1. Quick Farneback pass to estimate motion for decision-making
        # Convert Tensors to NumPy BGR for OpenCV processing
        img0_bgr_np = t_to_bgr_uint8(img0)
        img1_bgr_np = t_to_bgr_uint8(img1)
        gray0 = cv2.cvtColor(img0_bgr_np, cv2.COLOR_BGR2GRAY)
        gray1 = cv2.cvtColor(img1_bgr_np, cv2.COLOR_BGR2GRAY)
        
        # Use simplified Farneback parameters for fast motion estimation
        flow = cv2.calcOpticalFlowFarneback(gray0, gray1, None, 0.5, 1, 15, 2, 5, 1.2, 0)
        
        # Calculate the mean magnitude of the flow vector
        flow_mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        mean_flow_mag = np.mean(flow_mag)
        
        # 2. Decision Logic
        if mean_flow_mag < self.motion_threshold:
            # Low motion: Use Optical Flow (faster, simple movements are handled well)
            # This saves computation resources.
            return self.flow_classic(img0, img1, t)
        else:
            # High motion: Use RIFE (more accurate in complex/large motion scenarios)
            # RIFE's learned approach is superior when flow estimation is difficult.
            return self.rife.interpolate(img0.to(self.device), img1.to(self.device), t=t.item())