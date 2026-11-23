# shady nikooei

<<<<<<< HEAD
import math as m
import cv2
import numpy as np
from scipy.stats import gmean


def extract_keyframes(
    input_video_path: str,
    output_video_path: str = "final_summary_video_hsv.mp4",
    frame_rate: int = None,
):
  """
    Extracts keyframes from a video based on frame-to-frame differences in HSV color space.
    Applies a weighted geometric moving average filter to smooth the difference signal,
    selects keyframes based on a statistical threshold, and saves them into a summary video.

    Returns:
        np.ndarray: Array of extracted keyframe indices.
    """
  cap = cv2.VideoCapture(input_video_path)
  if not cap.isOpened():
    print("Error: Failed to open input video.")
    return np.array([])

  frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  if frame_rate is None:
    frame_rate = cap.get(cv2.CAP_PROP_FPS)

  diff_vector = []

  # Read the first frame and convert to HSV
  ret, prev_frame = cap.read()
  if not ret:
    print("Error: Failed to read the first frame.")
    cap.release()
    return np.array([])

  prev_hsv = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2HSV)

  # Compute Euclidean distance in HSV space between consecutive frames
  for _ in range(1, frame_count):
    ret, frame = cap.read()
    if not ret:
      break
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    diff = np.sqrt(
        np.sum((hsv.astype("float32") - prev_hsv.astype("float32")) ** 2)
    )
    diff_vector.append(diff)
    prev_hsv = hsv

  if not diff_vector:
    cap.release()
    return np.array([])

  # Ensure window_len is an odd integer
  window_len = int(m.sqrt(len(diff_vector)))
  if window_len % 2 == 0:
    window_len += 1
  window_len = max(3, window_len)

  # Construct symmetric geometric weight array
  half = window_len // 2
  a = 2
  r = 2
  left_w = [a * (r**i) for i in range(half)]
  center_w = [a * (r**half)]
  right_w = left_w[::-1]
  weights = np.array(left_w + center_w + right_w, dtype=np.float64)

  smoothed_diff = []

  for k in range(len(diff_vector)):
    center_val = max(diff_vector[k] * weights[half], 1e-6)

    # Left window
    num_left = min(k, half)
    left_vals = diff_vector[k - num_left : k]
    left_weights = weights[half - num_left : half]
    part1 = np.dot(left_vals, left_weights) if len(left_vals) > 0 else 1.0

    # Right window
    num_right = min(len(diff_vector) - k - 1, half)
    right_vals = diff_vector[k + 1 : k + 1 + num_right]
    right_weights = weights[half + 1 : half + 1 + num_right]
    part2 = np.dot(right_vals, right_weights) if len(right_vals) > 0 else 1.0

    total_weights = weights[half - num_left : half + 1 + num_right]
    weight_prod = np.prod(total_weights)

    # Avoid zero/negative values for geometric mean
    weight_parts = [max(part1, 1e-6), center_val, max(part2, 1e-6)]

    if weight_prod > 1:
      weighted_avg = gmean(weight_parts)
    else:
      weighted_avg = 0.0

    smoothed_diff.append(weighted_avg)

  # Statistical thresholding to select keyframes
  smoothed_diff = np.array(smoothed_diff)
  threshold = np.mean(smoothed_diff) + 1.5 * np.std(smoothed_diff)
  # Add +1 because diff_vector[i] corresponds to the transition into frame (i + 1)
  keyframe_indices = np.where(smoothed_diff >= threshold)[0] + 1

  # Reopen video to write extracted keyframes
  cap.release()
  cap = cv2.VideoCapture(input_video_path)

  width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

  out = cv2.VideoWriter(
      output_video_path,
      cv2.VideoWriter_fourcc(*"mp4v"),
      frame_rate,
      (width, height),
  )

  for idx in keyframe_indices:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
    ret, frame = cap.read()
    if ret:
      out.write(frame)

  cap.release()
  out.release()
  print("Summary video creation complete!")

  return keyframe_indices
=======
import cv2
import numpy as np
from scipy.stats import gmean
import math as m

def extract_keyframes(input_video_path: str, output_video_path: str = "final_summary_video_hsv.mp4", frame_rate: int = 5):
    """
    Extracts keyframes from a video based on frame-to-frame differences in HSV color space.
    Applies a weighted moving average filter to smooth the difference signal,
    then selects keyframes based on a statistical threshold and saves them into a new summary video.
    
    """

    # Read input video
    cap = cv2.VideoCapture(input_video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # importent for final comperessed vedio that maked from keyFrames extracted  (new)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)

    diff_vector = []
    
    # Read the first frame and convert to HSV
    ret, prev_frame = cap.read()
    if not ret:
        print("Failed to read the first frame.")
        cap.release()
        return

    prev_hsv = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2HSV)

    # Compute HSV difference between consecutive frames
    for _ in range(1, frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        diff = np.sqrt(np.sum((hsv.astype("float32") - prev_hsv.astype("float32")) ** 2))
        diff_vector.append(diff)
        prev_hsv = hsv

    # Weighted moving average smoothing
    # window_len = 11 , firsst method
    window_len = m.sqrt(len(diff_vector))    # Second method: order best last method and loss a bit accuresy But not too much. 



    # Constructing the weight array
    a = 2      # First sentence
    r = 2      # Common ratio
    n =  window_len // 2    # Number of sentences
    weights = [a * r**i for i in range(n)]

    # smoothing and padding
    half = window_len // 2
    smoothed_diff = []

    for k in range(len(diff_vector)):
        center_val = diff_vector[k] * weights[half]

        # Left window
        num_left = min(k, half)
        left_vals = diff_vector[k - num_left:k]
        left_weights = weights[half - num_left:half]
        part1 = np.dot(left_vals, left_weights) if left_vals else 1    # zero pading 2 4 8 16 32 1 1 1 1 1 ...
        part1 = np.dot(left_vals, left_weights) if left_vals else 1

        # Right window
        num_right = min(len(diff_vector) - k - 1, half)
        right_vals = diff_vector[k + 1:k + 1 + num_right]
        right_weights = weights[half + 1:half + 1 + num_right]
        part2 = np.dot(right_vals, right_weights) if right_vals else 1  # zero pading 00000012345
        part2 = np.dot(right_vals, right_weights) if right_vals else 1

        total_weights = weights[half - num_left:half + 1 + num_right]
        weight_mul = np.multiply(total_weights)

        weight_parts = [part1, center_val, part2]
        # smoothing weighted_averages from correlation

        if weight_mul > 1:
            #weighted_avg = (part1 + center_val + part2) / weight_sum     hesabiii
            weighted_avg = gmean(weight_parts) 
        else:
            weighted_avg = 0

        smoothed_diff.append(weighted_avg)

    # Thresholding to select keyframes
    smoothed_diff = np.array(smoothed_diff)
    threshold = np.mean(smoothed_diff) + 1.5 * np.std(smoothed_diff)
    keyframe_indices = np.where(smoothed_diff >= threshold)[0]

    


    # Reopen video to read keyframes for saving
    cap.release()
    cap = cv2.VideoCapture(input_video_path)
    
    out = cv2.VideoWriter(
        output_video_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        frame_rate,
        (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    )

    for idx in keyframe_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        # BY defualt is BGR in openCV
        ret, frame = cap.read()
        if ret:
            out.write(frame)

    cap.release()
    out.release()
    print("Video creation complete!")
>>>>>>> 7fe33a0f352f101414cd6c86211a266c0331ef77
