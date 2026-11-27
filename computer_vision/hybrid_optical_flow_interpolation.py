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

def first_Preprocessing(frame_a, frame_c):
    
    gray_a = cv2.cvtColor(frame_a, cv2.COLOR_RGB2GRAY)
    gray_c = cv2.cvtColor(frame_c, cv2.COLOR_RGB2GRAY)

    # 1. Calculate Optical Flow (Farneback) with Optimized Parameters
    # IMPROVEMENT: Increased levels, winsize, and iterations for better stability/accuracy
    #              Reduced poly_n for better local detail preservation.
    flow = cv2.calcOpticalFlowFarneback(
        gray_a, gray_c,
        None,
        pyr_scale=0.5,
        levels=5,       # Increased from 3 for better handling of larger displacements
        winsize=25,     # Increased from 15 for better stability (less noise sensitivity)
        iterations=5,   # Increased from 3 for better convergence
        poly_n=3,       # Decreased from 5 for higher local detail retention
        poly_sigma=1.2,
        flags=0
    )

    
    # calculate size of distance vector
    mag, angle = cv2.cartToPolar(
    flow[:,:,0],
    flow[:,:,1]
    )

    # calculate thereshold
    #thereshold = int(np.mean(mag) + 1.5*np.std(mag))

    # An approach for storing indices greater than the threshold 
    #g_vector = []

    # number of greater elements
    #cnt = 0
    #for i in range(1,len(flow[:,:,0])):
    #    if(mag > thereshold):
    #        g_vector[i] = 1
    #        cnt += 1
    #    else:
    #        g_vector[i] = 0
    # Calculate threshold based on mean and standard deviation of magnitude
    threshold = int(np.mean(mag) + 1.5 * np.std(mag))

    # Store binary indicators for elements greater than the threshold using vectorized operations
    g_vector = (mag > threshold).astype(int).tolist()

    # Count the number of elements exceeding the threshold
    cnt = sum(g_vector)
    num_graterItem = len(g_vector)
    # Percentage of number frames grater than thereshold
    Percentage_frequency1 =  (cnt/num_graterItem)*100
    output_items = [gray_a, mag, flow[:,:,0], flow[:,:,1], thereshold, Percentage_frequency1]
    return output_items
    


def hybrid_optical_flow_interpolation(frame_a, frame_c, fix_border=True):
    """
    Interpolates a frame between two input frames using optimized classical optical flow (Farneback).
    The method is enhanced with tuned parameters, CUBIC interpolation, and denoising 
    to maximize visual quality (SSIM/PSNR) for low-motion scenes.

    Parameters:
        frame_a (np.ndarray): First RGB frame (uint8).
        frame_c (np.ndarray): Second RGB frame (uint8).
        fix_border (bool): Whether to apply RegularGridInterpolator on remap edge artifacts.

    Returns:
        interpolated (np.ndarray): Interpolated frame (uint8).
    """ 
    info = first_Preprocessing(frame_a, frame_c)
    
    gray_a = info[0]
    mag = info[1]
    flow0 = info[2]
    flow1 = info[3]
    thereshold= info[4]
    Percentage_frequency1 = info[5]


    thereshold_aray = []
    thereshold_aray.append(thereshold)
    # numer of frames
    new_frames = 0
    n = 2 # first number
    if(Percentage_frequency1 >= 50):
        #while not(mag - thereshold < 0):
        for idx in range(n):
            mask = (mag - thereshold_aray[idx]) >= 0
            if(np.sum(mask) >= mask.size / 2):
                new_frames +=1
                h, w = gray_a.shape
                X, Y = np.meshgrid(np.arange(w), np.arange(h))

                half_vx = flow0 * 0.5
                half_vy = flow1 * 0.5

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

                    # IMPROVEMENT: Upgraded interpolation to CUBIC for better visual quality and edge sharpness.
                    warped_a[:, :, k] = cv2.remap(channel_a, map_ax, map_ay, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
                    warped_c[:, :, k] = cv2.remap(channel_c, map_cx, map_cy, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)

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
                
                # 2. IMPROVEMENT: Post-processing Denoising (Median Blur)
                # Reduces flow noise and can boost PSNR/SSIM, especially for low-motion scenes.
                blended_uint8 = np.clip(blended, 0, 255).astype(np.uint8)
                blended_denoised = cv2.medianBlur(blended_uint8, 3) 
            n = n + new_frames  
            hybrid_optical_flow_interpolation(frame_a,blended_denoised)
            hybrid_optical_flow_interpolation(blended_denoised,frame_c)  
    else:
        return blended_denoised

