import torch
import cv2
import numpy as np

# Import project modules with precise addressing
from deepvision.models.rife_wrapper import RIFEInterpolator
from deepvision.models.optical_flow_wrapper import OpticalFlowInterpolator
from deepvision.pipelines.interpolate import t_to_bgr_uint8, bgr_uint8_to_t

class AdaptiveInterpolator(torch.nn.Module):
    """
    Adaptive Interpolator:
    A hybrid class that intelligently switches between the classical method (Optical Flow)
    and the deep learning method (RIFE) based on the estimated motion magnitude within the scene.
    """
    def __init__(self, rife_impl=None, rife_ckpt=None, motion_threshold: float = 5.0):
        super().__init__()
        
        # 1. Load the Deep Learning Model (RIFE)
        # This model is utilized for complex scenes and rapid movements.
        self.rife = RIFEInterpolator().load_pretrained(impl=rife_impl, ckpt_path=rife_ckpt)
        
        # 2. Load the Classical Model (Optical Flow)
        # This model is used for low-motion and simple scenes (higher speed/efficiency).
        self.flow_classic = OpticalFlowInterpolator()
        
        # Decision Threshold (Switch to RIFE if motion exceeds this value)
        self.motion_threshold = motion_threshold
        
        # Set the processing device (CPU or GPU) based on RIFE model settings
        self.device = self.rife.device if self.rife.device else 'cpu'
        
        print(f"Adaptive switch threshold set to: {self.motion_threshold} (Mean Flow Magnitude)")

    def forward(self, img0: torch.Tensor, img1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Main Decision Logic:
        First estimates the motion, then executes the appropriate model.
        """
        # --- Step 1: Rapid Motion Estimation ---
        
        # Convert graphics tensors to NumPy format for OpenCV processing
        img0_bgr_np = t_to_bgr_uint8(img0)
        img1_bgr_np = t_to_bgr_uint8(img1)
        
        # Convert to Grayscale for faster optical flow calculation
        gray0 = cv2.cvtColor(img0_bgr_np, cv2.COLOR_BGR2GRAY)
        gray1 = cv2.cvtColor(img1_bgr_np, cv2.COLOR_BGR2GRAY)
        
        # Use Gunnar Farneback algorithm to calculate the optical flow field
        # (Parameters are tuned for high speed)
        flow = cv2.calcOpticalFlowFarneback(gray0, gray1, None, 0.5, 1, 15, 2, 5, 1.2, 0)
        
        # Calculate the magnitude of the motion vector for each pixel
        flow_mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        
        # Calculate the mean motion magnitude across the entire image
        mean_flow_mag = np.mean(flow_mag)
        
        # --- Step 2: Decision and Execution ---
        
        if mean_flow_mag < self.motion_threshold:
            # Case 1: Low Motion
            # Use the Classical Method -> Saves resources and time
            return self.flow_classic(img0, img1, t)
        else:
            # Case 2: High Motion
            # Use the RIFE Neural Network -> Higher accuracy for non-linear movements
            # Note: Inputs must be moved to the device (GPU)
            return self.rife.interpolate(img0.to(self.device), img1.to(self.device), t=t.item())

    def interpolate(self, img0, img1, t=0.5):
        """
        Wrapper Method:
        This function is called by the 'infer_video.py' script.
        Its task is to convert the scalar time 't' into the appropriate tensor format and call forward().
        """
        # Convert the float time value to a PyTorch tensor
        t_tensor = torch.tensor([t], dtype=torch.float32, device=self.device)
        
        # specific call to the main processing function
        return self.forward(img0, img1, t_tensor)