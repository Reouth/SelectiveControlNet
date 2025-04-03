import os
import glob
from PIL import Image
import numpy as np
import cv2

def extract_two_frames(video_path, frame_indices=[0, 1]):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for idx in frame_indices:
        if idx >= total:
            raise ValueError(f"Requested frame {idx} exceeds total frames {total}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
    cap.release()
    return frames

def frames_to_video(frames, output_path, fps=2):

    if not frames:
        raise ValueError("No frames provided")

    # Convert first frame to get size
    first_frame = frames[0].convert("RGB")
    width, height = first_frame.size

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    for frame in frames:
        frame = frame.convert("RGB")  # Ensure no alpha
        frame_np = np.array(frame)
        frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)

    out.release()

def find_latest_output_video(base_dir="outputs"):
    video_files = glob.glob(os.path.join(base_dir, "**", "video.mp4"), recursive=True)
    video_files.sort(key=os.path.getmtime, reverse=True)
    return video_files[0] if video_files else None