from PIL import Image
import numpy as np
from src.image_utils import validate_image_sizes
from src.masking import normalize_mask

def blend_images(fg_np, bg_np, mask_np):
    return fg_np * mask_np + bg_np * (1 - mask_np)

def merge_images(foreground_image, background_image, mask):
    """
    Merge foreground and background using a binary mask.
    """
    try:
        validate_image_sizes(foreground_image, background_image, mask)

        fg = foreground_image.convert("RGB")
        bg = background_image.convert("RGB")

        fg_np = np.array(fg).astype(np.float32)
        bg_np = np.array(bg).astype(np.float32)
        mask_np = normalize_mask(mask)

        blended_np = blend_images(fg_np, bg_np, mask_np)
        return Image.fromarray(np.uint8(blended_np))

    except Exception as e:
        print(f"[merge_images] Error: {e}")
        return None
