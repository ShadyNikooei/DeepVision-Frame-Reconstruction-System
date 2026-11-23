# shady nikooei

from keyframe_extraction import extract_keyframes
from make_incomplete_video import create_incomplete_video
from reconstruct_full_video import reconstruct_video_with_interpolation


def main():
  original_video_path = "input_video.mp4"

  # Step 1: Extract keyframes
  summary_video_path = "summary_keyframes.mp4"
  print("Step 1: Extracting keyframes...")
  keyframes = extract_keyframes(
      original_video_path, summary_video_path, frame_rate=5
  )
  print(f"{len(keyframes)} keyframes extracted.")

  # Step 2: Create incomplete video by dropping every other frame
  incomplete_video_path = "incomplete_video.mp4"
  print("\nStep 2: Creating incomplete video...")
  num_written = create_incomplete_video(
      original_video_path, incomplete_video_path, frame_rate=15
  )
  print(f"Incomplete video created with {num_written} frames.")

  # Step 3: Reconstruct full video with interpolated frames (restoring 30 FPS)
  reconstructed_video_path = "reconstructed_video.mp4"
  print("\nStep 3: Reconstructing full video with interpolation...")
  reconstruct_video_with_interpolation(
      incomplete_video_path, reconstructed_video_path, target_fps=30
  )
  print("Final reconstructed video saved.")


if __name__ == "__main__":
  main()