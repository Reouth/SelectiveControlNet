from PIL import Image
import numpy as np
import cv2

def load_and_resize_image(path, size=(512, 512)):
    return Image.open(path).convert("RGB").resize(size)


def create_masked_image(image, mask):
    if image.size != mask.size:
        raise ValueError("Image and mask sizes do not match.")

    mask_np = np.array(mask)

    if len(mask_np.shape) == 3:
        mask_np = mask_np[:, :, 0]
    mask_np = (mask_np / np.max(mask_np) * 255).astype(np.uint8)
    mask = Image.fromarray(mask_np).convert("L")

    return Image.composite(image, Image.new("RGB", image.size, (0, 0, 0)), mask)

def validate_same_size(*images):

    if not images:
        raise ValueError("At least one image must be provided.")

    first_size = images[0].size
    for i, img in enumerate(images[1:], start=2):
        if img.size != first_size:
            raise ValueError(f"Image #{i} has size {img.size}, expected {first_size}.")

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

