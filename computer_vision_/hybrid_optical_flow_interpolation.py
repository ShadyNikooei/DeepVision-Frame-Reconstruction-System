# shady nikooei

import cv2
import numpy as np
from scipy.interpolate import RegularGridInterpolator

def hybrid_optical_flow_interpolation(frame_a, frame_c, fix_border=True):
    """
    Interpolates a frame between two input frames using optical flow (Farneback).
    Uses fast remap for most of the image, and switches to RegularGridInterpolator only
    where remap gives unreliable results (e.g., near borders or undefined pixels).

    Parameters:
        frame_a (np.ndarray): First RGB frame (uint8).
        frame_c (np.ndarray): Second RGB frame (uint8).
        fix_border (bool): Whether to apply RegularGridInterpolator on remap edge artifacts.

    Returns:
        interpolated (np.ndarray): Interpolated frame (uint8).
    """

    gray_a = cv2.cvtColor(frame_a, cv2.COLOR_RGB2GRAY)
    gray_c = cv2.cvtColor(frame_c, cv2.COLOR_RGB2GRAY)

    flow = cv2.calcOpticalFlowFarneback(
        gray_a, gray_c,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0
    )

    h, w = gray_a.shape
    X, Y = np.meshgrid(np.arange(w), np.arange(h))

    half_vx = flow[..., 0] / 2
    half_vy = flow[..., 1] / 2

    warped_a = np.zeros_like(frame_a, dtype=np.float32)
    warped_c = np.zeros_like(frame_c, dtype=np.float32)

    for k in range(3):
        channel_a = frame_a[:, :, k].astype(np.float32)
        channel_c = frame_c[:, :, k].astype(np.float32)

        # Fast warping with cv2.remap
        map_ax = (X + half_vx).astype(np.float32)
        map_ay = (Y + half_vy).astype(np.float32)
        map_cx = (X - half_vx).astype(np.float32)
        map_cy = (Y - half_vy).astype(np.float32)

        warped_a[:, :, k] = cv2.remap(channel_a, map_ax, map_ay, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        warped_c[:, :, k] = cv2.remap(channel_c, map_cx, map_cy, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

        if fix_border:
            # Detect remap edge artifacts (near-zero or constant flat values)
            mask_a = (warped_a[:, :, k] < 1)
            mask_c = (warped_c[:, :, k] < 1)

            if np.any(mask_a):
                interpolator_a = RegularGridInterpolator(
                    (np.arange(h), np.arange(w)),
                    channel_a,
                    bounds_error=False,
                    fill_value=0
                )
                coords_a = np.stack([map_ay[mask_a], map_ax[mask_a]], axis=-1)
                warped_a[:, :, k][mask_a] = interpolator_a(coords_a)

            if np.any(mask_c):
                interpolator_c = RegularGridInterpolator(
                    (np.arange(h), np.arange(w)),
                    channel_c,
                    bounds_error=False,
                    fill_value=0
                )
                coords_c = np.stack([map_cy[mask_c], map_cx[mask_c]], axis=-1)
                warped_c[:, :, k][mask_c] = interpolator_c(coords_c)

    # Blend the warped frames
    blended = 0.5 * warped_a + 0.5 * warped_c
    return np.clip(blended, 0, 255).astype(np.uint8)
