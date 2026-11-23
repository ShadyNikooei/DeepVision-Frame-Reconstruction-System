"""
Utility helpers for reading/writing videos and working with FPS.
All functions are dependency-light (OpenCV + NumPy) and robust to basic errors.
"""
from typing import List
import cv2
import numpy as np

def read_video_frames(path: str) -> List[np.ndarray]:
    """Read all frames from a video file into a list of BGR uint8 images.

    Raises:
        RuntimeError: if the video cannot be opened or has no frames.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame is None:
            break
        frames.append(frame)
    cap.release()
    if len(frames) == 0:
        raise RuntimeError("No frames were read; check codec/path")
    return frames

def write_video_frames(path: str, frames: List[np.ndarray], fps: float):
    """Write a list of BGR uint8 frames to a video file.

    Notes:
        * We default to mp4v fourcc for wide compatibility.
        * All frames are resized to the size of the first frame if mismatched.
    """
    if len(frames) == 0:
        raise ValueError("No frames to write")
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(path, fourcc, fps, (w, h))
    if not out.isOpened():
        raise RuntimeError(f"Cannot open writer for: {path}")
    for f in frames:
        if f.shape[0] != h or f.shape[1] != w:
            f = cv2.resize(f, (w, h), interpolation=cv2.INTER_CUBIC)
        out.write(f)
    out.release()

def get_fps(path: str) -> float:
    """Return the FPS of a video, defaulting to 30.0 if not reported."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    return float(fps)
