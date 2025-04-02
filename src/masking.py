from rembg import remove
from PIL import Image, ImageOps
import numpy as np

def get_foreground_mask(image):
    return remove(image, only_mask=True)

def get_background_mask(image):
    foreground_mask = get_foreground_mask(image)
    return ImageOps.invert(foreground_mask)

def get_foreground_image(image):
    mask = get_foreground_mask(image)
    return Image.composite(image, Image.new("RGB", image.size, (0, 0, 0)), mask)

def get_background_image(image):
    mask = get_background_mask(image)
    return Image.composite(image, Image.new("RGB", image.size, (0, 0, 0)), mask)

def normalize_mask(mask):
    """
    Convert mask to grayscale and normalize to [0, 1].
    """
    mask = mask.convert("L")
    mask_np = np.array(mask).astype(np.float32)
    if np.max(mask_np) != 0:
        mask_np /= np.max(mask_np)
    mask_np = np.expand_dims(mask_np, axis=-1)
    return mask_np

def get_foreground_images(frames):
    return [get_foreground_image(frame) for frame in frames]