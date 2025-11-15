# shady nikooei

import cv2
import numpy as np

def extract_keyframes(input_video_path: str, output_video_path: str = "final_summary_video_hsv.mp4", frame_rate: int = 5):
    """
    Extracts keyframes from a video based on frame-to-frame differences in HSV color space.
    Applies a weighted moving average filter to smooth the difference signal,
    then selects keyframes based on a statistical threshold and saves them into a new summary video.
    
    """

    # Read input video
    cap = cv2.VideoCapture(input_video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

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
    window_len = 11
    weights = [0.05, 0.05, 0.05, 0.1, 0.15, 0.2, 0.15, 0.1, 0.05, 0.05, 0.05]
    half = window_len // 2
    smoothed_diff = []

    for k in range(len(diff_vector)):
        center_val = diff_vector[k] * weights[half]

        # Left window
        num_left = min(k, half)
        left_vals = diff_vector[k - num_left:k]
        left_weights = weights[half - num_left:half]
        part1 = np.dot(left_vals, left_weights) if left_vals else 0

        # Right window
        num_right = min(len(diff_vector) - k - 1, half)
        right_vals = diff_vector[k + 1:k + 1 + num_right]
        right_weights = weights[half + 1:half + 1 + num_right]
        part2 = np.dot(right_vals, right_weights) if right_vals else 0

        total_weights = weights[half - num_left:half + 1 + num_right]
        weight_sum = np.sum(total_weights)

        if weight_sum > 0:
            weighted_avg = (part1 + center_val + part2) / weight_sum
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
        ret, frame = cap.read()
        if ret:
            out.write(frame)

    cap.release()
    out.release()
    print("Video creation complete!")
