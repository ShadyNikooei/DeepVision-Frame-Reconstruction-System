# shady nikooei

import cv2
from hybrid_optical_flow_interpolation import hybrid_optical_flow_interpolation

def reconstruct_video(input_video_path: str, output_video_path: str = "complete_video.mp4"):
    """
    Reconstructs a video by interpolating intermediate frames using optical flow.
    Assumes the input video is missing every other frame.

    Parameters:
        input_video_path (str): Path to the incomplete video file.
        output_video_path (str): Path to save the reconstructed video.
    """

    # Open the input video
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Failed to open input video.")
        return

    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames = []

    print("Reading frames into memory...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    print(f"Loaded {len(frames)} frames.")

    # Prepare output writer with higher framerate
    out = cv2.VideoWriter(
        output_video_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        frame_rate * 1.5,
        (width, height)
    )

    print("Reconstructing full video with interpolated frames...")
    for i in range(len(frames) - 1):
        out.write(frames[i])
        interpolated = hybrid_optical_flow_interpolation(frames[i], frames[i + 1])
        out.write(interpolated)

    # Write the last original frame
    out.write(frames[-1])
    out.release()

    print("Video reconstruction complete.")
