# shady nikooei

import cv2
from hybrid_optical_flow_interpolation import hybrid_optical_flow_interpolation

<<<<<<< HEAD

def reconstruct_video_with_interpolation(
    input_video_path: str,
    output_video_path: str = "reconstructed_video.mp4",
    target_fps: float = None,
):
  """
    Reconstructs a full video by interpolating intermediate frames between consecutive
    frames of an incomplete video using hybrid optical flow.
=======
def reconstruct_video(input_video_path: str, output_video_path: str = "complete_video.mp4"):
    """
    Reconstructs a video by interpolating intermediate frames using optical flow.
    Assumes the input video is video that compresed with keyFrame Extraction  # missing every other frame.
>>>>>>> 7fe33a0f352f101414cd6c86211a266c0331ef77

    Parameters:
        input_video_path (str): Path to the incomplete video file.
        output_video_path (str): Path to save the reconstructed video.
<<<<<<< HEAD
        target_fps (float, optional): Output frame rate. Defaults to 2x input FPS.
    """
  cap = cv2.VideoCapture(input_video_path)
  if not cap.isOpened():
    print("Error: Failed to open input video.")
    return

  input_fps = cap.get(cv2.CAP_PROP_FPS)
  output_fps = target_fps if target_fps is not None else (input_fps * 2)
  width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
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

  if len(frames) < 2:
    print("Error: Not enough frames to perform interpolation.")
    return

  out = cv2.VideoWriter(
      output_video_path,
      cv2.VideoWriter_fourcc(*"mp4v"),
      output_fps,
      (width, height),
  )

  print("Reconstructing full video with interpolated frames...")
  for i in range(len(frames) - 1):
    out.write(frames[i])
    interpolated = hybrid_optical_flow_interpolation(frames[i], frames[i + 1])
    out.write(interpolated)

  # Write the final original frame
  out.write(frames[-1])
  out.release()
  print("Video reconstruction complete.")
=======
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

    # Prepare output writer  #with higher framerate frame_rate * 2
    out = cv2.VideoWriter(
        output_video_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        frame_rate,   
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
>>>>>>> 7fe33a0f352f101414cd6c86211a266c0331ef77
