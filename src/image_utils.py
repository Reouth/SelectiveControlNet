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
