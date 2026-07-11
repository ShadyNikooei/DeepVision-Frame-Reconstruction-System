# shady nikooei

import cv2

def create_incomplete_video(input_video_path: str, output_video_path: str = "incomplete_video.mp4", frame_rate: int = 20):
    """
    Creates an incomplete video by keeping only even-indexed frames from the input video.
    This can be used to test interpolation or frame-reconstruction algorithms.

    """

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Failed to open the input video.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out = cv2.VideoWriter(
        output_video_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        frame_rate,
        (width, height)
    )

    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        # Keep only even-indexed frames (i.e., frame numbers 0, 2, 4, ...)
        if i % 2 == 0:  # MATLAB indices start at 1, Python at 0
            out.write(frame)

    cap.release()
    out.release()
    print("Incomplete video created successfully.")
