"""
RIFE wrapper with two interchangeable backends:
  1) DummyAverageInterpolator – for pipeline sanity-check (no DL)
  2) Real RIFE – plug any third-party RIFE implementation without modifying the pipeline

USAGE (Real RIFE):
    from deepvision.models.rife_wrapper import RIFEInterpolator
    rife = RIFEInterpolator().load_pretrained(
        impl="some_repo.rife:RIFE",   # module:Class path
        ckpt_path="weights/rife.pth"  # optional; depends on the chosen repo
    )

The wrapper standardizes the interface to `.interpolate(img0, img1, t)`.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Callable
import importlib
import torch
import torch.nn as nn

class DummyAverageInterpolator(nn.Module):
    """A non-DL baseline to validate the pipeline.
    It simply averages the two input frames in RGB space.
    DO NOT use for results – just to check the I/O path works.
    """
    def forward(self, img0: torch.Tensor, img1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # img0/img1: (1,3,H,W) in [0,1]; t: (1,) in [0,1]
        # A trivial convex blend – independent of t to keep it simple
        return (img0 + img1) * 0.5

@dataclass
class RIFEInterpolator:
    device: Optional[str] = None

    def __post_init__(self):
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[nn.Module] = None

    def load_pretrained(self, impl: Optional[str] = None, ckpt_path: Optional[str] = None) -> "RIFEInterpolator":
        """Load a real RIFE model (preferred) or fall back to the dummy interpolator.

        Args:
            impl: A string like "package.sub.module:ClassName" that constructs the model.
            ckpt_path: Optional path to a checkpoint file (depends on the chosen implementation).
        """
        if impl is None:
            # Fallback: dummy interpolator (so the pipeline runs even without DL)
            self.model = DummyAverageInterpolator().to(self.device).eval()
            return self

        # Dynamically import the user's chosen RIFE implementation
        module_name, class_name = impl.split(":")
        mod = importlib.import_module(module_name)
        ctor: Callable = getattr(mod, class_name)
        model = ctor()

        # Try to load weights if provided (implementation-dependent)
        if ckpt_path is not None:
            state = torch.load(ckpt_path, map_location=self.device)
            # Many repos save as {"state_dict": ...}; support both formats
            state_dict = state.get("state_dict", state)
            model.load_state_dict(state_dict, strict=False)

        self.model = model.to(self.device).eval()
        return self

    @torch.inference_mode()
    def interpolate(self, img0: torch.Tensor, img1: torch.Tensor, t: float = 0.5) -> torch.Tensor:
        """Interpolate a single intermediate frame between img0 and img1.

        Args:
            img0, img1: Tensors with shape (1,3,H,W), RGB, float32, range [0,1]
            t: a float in [0,1] where 0.5 is the middle frame
        Returns:
            Tensor with shape (1,3,H,W), RGB in [0,1]
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_pretrained() first.")
        t_tensor = torch.tensor([t], dtype=torch.float32, device=self.device)
        img0 = img0.to(self.device)
        img1 = img1.to(self.device)
        out = self.model(img0, img1, t_tensor)
        # Ensure valid range
        return out.clamp(0.0, 1.0)
