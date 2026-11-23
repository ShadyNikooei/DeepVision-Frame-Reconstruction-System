# shady nikooei

import cv2
import numpy as np
from scipy.interpolate import RegularGridInterpolator


def analyze_motion(frame_a: np.ndarray, frame_c: np.ndarray):
  """
    Converts BGR frames to grayscale, computes dense optical flow (Farneback),
    and calculates motion statistics to determine the significance of movement.
    """
  # OpenCV reads frames in BGR format by default
  gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
  gray_c = cv2.cvtColor(frame_c, cv2.COLOR_BGR2GRAY)

  # Calculate Optical Flow (Farneback) with optimized parameters
  flow = cv2.calcOpticalFlowFarneback(
      gray_a,
      gray_c,
      None,
      pyr_scale=0.5,
      levels=5,  # Better handling of larger displacements
      winsize=25,  # Higher stability and less noise sensitivity
      iterations=5,  # Better convergence
      poly_n=3,  # Higher local detail retention
      poly_sigma=1.2,
      flags=0,
  )

  # Calculate magnitude of displacement vectors
  mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

  # Calculate statistical motion threshold
  threshold = np.mean(mag) + 1.5 * np.std(mag)

  # Percentage of pixels with motion magnitude greater than the threshold
  high_motion_ratio = (np.sum(mag > threshold) / mag.size) * 100.0

  return gray_a, flow, threshold, high_motion_ratio


def hybrid_optical_flow_interpolation(
    frame_a: np.ndarray, frame_c: np.ndarray, fix_border: bool = True
) -> np.ndarray:
  """
    Interpolates an intermediate frame between two input frames using optimized 
    classical optical flow (Farneback), CUBIC remapping, edge artifact correction, 
    and post-processing denoising.

    Parameters:
        frame_a (np.ndarray): First BGR frame (uint8).
        frame_c (np.ndarray): Second BGR frame (uint8).
        fix_border (bool): Whether to apply RegularGridInterpolator on remap edge artifacts.

    Returns:
        np.ndarray: Interpolated BGR frame (uint8).
    """
  gray_a, flow, _, high_motion_ratio = analyze_motion(frame_a, frame_c)

  # If there is almost zero motion between frames, simple linear blending is faster and cleaner
  if high_motion_ratio < 0.5:
    blended = 0.5 * frame_a.astype(np.float32) + 0.5 * frame_c.astype(
        np.float32
    )
    return np.clip(blended, 0, 255).astype(np.uint8)

  h, w = gray_a.shape
  X, Y = np.meshgrid(np.arange(w), np.arange(h))

  half_vx = flow[..., 0] * 0.5
  half_vy = flow[..., 1] * 0.5

  warped_a = np.zeros_like(frame_a, dtype=np.float32)
  warped_c = np.zeros_like(frame_c, dtype=np.float32)

  # Coordinate maps for forward and backward warping
  map_ax = (X + half_vx).astype(np.float32)
  map_ay = (Y + half_vy).astype(np.float32)
  map_cx = (X - half_vx).astype(np.float32)
  map_cy = (Y - half_vy).astype(np.float32)

  for k in range(3):
    channel_a = frame_a[:, :, k].astype(np.float32)
    channel_c = frame_c[:, :, k].astype(np.float32)

    # Fast warping with cv2.remap using CUBIC interpolation for edge sharpness
    warped_a[:, :, k] = cv2.remap(
        channel_a,
        map_ax,
        map_ay,
        interpolation=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT,
    )
    warped_c[:, :, k] = cv2.remap(
        channel_c,
        map_cx,
        map_cy,
        interpolation=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT,
    )

    if fix_border:
      # Detect remap edge artifacts (near-zero or out-of-bound values)
      mask_a = warped_a[:, :, k] < 1
      mask_c = warped_c[:, :, k] < 1

      if np.any(mask_a):
        interpolator_a = RegularGridInterpolator(
            (np.arange(h), np.arange(w)),
            channel_a,
            bounds_error=False,
            fill_value=0,
        )
        coords_a = np.stack([map_ay[mask_a], map_ax[mask_a]], axis=-1)
        warped_a[:, :, k][mask_a] = interpolator_a(coords_a)

      if np.any(mask_c):
        interpolator_c = RegularGridInterpolator(
            (np.arange(h), np.arange(w)),
            channel_c,
            bounds_error=False,
            fill_value=0,
        )
        coords_c = np.stack([map_cy[mask_c], map_cx[mask_c]], axis=-1)
        warped_c[:, :, k][mask_c] = interpolator_c(coords_c)

  # Blend the warped frames equally
  blended = 0.5 * warped_a + 0.5 * warped_c

  # Post-processing Denoising (Median Blur) to reduce flow artifacts
  blended_uint8 = np.clip(blended, 0, 255).astype(np.uint8)
  blended_denoised = cv2.medianBlur(blended_uint8, 3)

  return blended_denoised