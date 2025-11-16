import torch
import cv2
import numpy as np
from hybrid_optical_flow_interpolation import hybrid_optical_flow_interpolation
# Import existing conversion utilities from the deep learning pipeline
from deepvision.pipelines.interpolate import t_to_bgr_uint8, bgr_uint8_to_t 

class OpticalFlowInterpolator(torch.nn.Module):
    """
    A PyTorch-compatible wrapper for the custom hybrid Optical Flow interpolation function.
    
    This class enables the use of the classic computer vision method (written in NumPy/OpenCV) 
    within the deep learning pipeline, allowing for adaptive switching and comparison.
    """
    def __init__(self, fix_border: bool = True):
        super().__init__()
        self.fix_border = fix_border
        # Set device to 'cpu' as the underlying function uses NumPy/OpenCV
        self.device = "cpu" 

    def forward(self, img0: torch.Tensor, img1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Performs interpolation by calling the underlying NumPy/OpenCV function.

        Args:
            img0, img1: Input frames as Tensors (1, C, H, W), RGB, float [0, 1].
            t: Time parameter (ignored here, as the classic method only does t=0.5).
            
        Returns:
            interpolated: Output frame as a Tensor (1, C, H, W), RGB, float [0, 1].
        """
        # Step 1: Convert PyTorch Tensor (RGB [0,1]) to NumPy array (BGR uint8)
        img0_bgr_np = t_to_bgr_uint8(img0)
        img1_bgr_np = t_to_bgr_uint8(img1)
        
        # Step 2: Convert BGR (OpenCV default) to RGB, as your hybrid function expects RGB
        frame_a = cv2.cvtColor(img0_bgr_np, cv2.COLOR_BGR2RGB)
        frame_c = cv2.cvtColor(img1_bgr_np, cv2.COLOR_BGR2RGB)

        # Step 3: Call the core Optical Flow implementation
        interpolated_rgb_np = hybrid_optical_flow_interpolation(frame_a, frame_c, self.fix_border)
        
        # Step 4: Convert back to BGR, then to PyTorch Tensor (RGB [0, 1])
        # The deep learning pipeline expects the final output in the Tensor format.
        return bgr_uint8_to_t(cv2.cvtColor(interpolated_rgb_np, cv2.COLOR_RGB2BGR))

    def load_pretrained(self, **kwargs) -> "OpticalFlowInterpolator":
        """Method required for compatibility with the existing RIFE-based pipeline setup."""
        return self