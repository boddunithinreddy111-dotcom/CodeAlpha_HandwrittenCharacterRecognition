"""
Image Processing Pipeline for Handwritten Character Recognition.
Performs grayscale conversion, background polarity auto-detection,
bounding-box cropping, center-of-mass alignment, and normalization.
"""

from pathlib import Path
from typing import Union
import numpy as np
from PIL import Image, ImageOps
from scipy.ndimage import center_of_mass, shift


def preprocess_image_array(
    image_input: Union[np.ndarray, Image.Image, str, Path],
    target_size: int = 28
) -> np.ndarray:
    """
    Takes an input image (PIL Image, numpy array, str or Path),
    applies image processing, and returns a normalized (1, 784) feature array.
    """
    # 1. Load or convert to PIL Image in Grayscale ('L')
    if isinstance(image_input, (str, Path)):
        img = Image.open(str(image_input)).convert("L")
    elif isinstance(image_input, Image.Image):
        img = image_input.convert("L")
    elif isinstance(image_input, np.ndarray):
        if image_input.ndim == 3 and image_input.shape[2] == 4:  # RGBA
            # Extract alpha channel or convert
            alpha = image_input[:, :, 3]
            rgb = image_input[:, :, :3]
            # If drawn on transparent canvas, use alpha as stroke mask
            if np.max(alpha) > 0 and np.min(alpha) == 0:
                img_arr = alpha
            else:
                img_arr = np.mean(rgb, axis=2).astype(np.uint8)
            img = Image.fromarray(img_arr)
        elif image_input.ndim == 3:
            img = Image.fromarray(np.mean(image_input, axis=2).astype(np.uint8))
        else:
            img = Image.fromarray(image_input.astype(np.uint8))
    else:
        raise ValueError("Unsupported image input type.")

    # 2. Auto-detect background polarity (EMNIST/MNIST has black background & white stroke)
    arr = np.array(img, dtype=np.float32)
    # Check if corners are light (white paper with black pen)
    corners = [arr[0, 0], arr[0, -1], arr[-1, 0], arr[-1, -1]]
    if np.mean(corners) > 127:
        # Invert so background is black and stroke is white
        img = ImageOps.invert(img)
        arr = np.array(img, dtype=np.float32)

    # 3. Thresholding / Denoising (Zero out background noise below 30)
    arr[arr < 30.0] = 0.0

    # 4. Bounding box cropping (tight crop around the stroke)
    coords = np.argwhere(arr > 30.0)
    if len(coords) == 0:
        # Empty drawing - return zero array
        return np.zeros((1, target_size * target_size), dtype=np.float32)

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    cropped_arr = arr[y0:y1, x0:x1]
    cropped_img = Image.fromarray(cropped_arr.astype(np.uint8))

    # 5. Maintain aspect ratio and fit into a box with 4-pixel padding
    content_size = 20  # Leaves 4 pixels on each border of the 28x28 canvas
    w, h = cropped_img.size
    if w > h:
        new_w = content_size
        new_h = max(1, int(round((h * content_size) / w)))
    else:
        new_h = content_size
        new_w = max(1, int(round((w * content_size) / h)))

    resized_stroke = cropped_img.resize((new_w, new_h), Image.Resampling.BILINEAR)

    # Place centered onto a 28x28 black canvas
    canvas = Image.new("L", (target_size, target_size), color=0)
    paste_x = (target_size - new_w) // 2
    paste_y = (target_size - new_h) // 2
    canvas.paste(resized_stroke, (paste_x, paste_y))

    canvas_arr = np.array(canvas, dtype=np.float32)

    # 6. Center-of-Mass Alignment (Standard MNIST/EMNIST normalization)
    cy, cx = center_of_mass(canvas_arr)
    if not (np.isnan(cy) or np.isnan(cx)):
        shift_y = int(np.round(14.0 - cy))
        shift_x = int(np.round(14.0 - cx))
        # Keep shift within safe bounds
        shift_y = np.clip(shift_y, -3, 3)
        shift_x = np.clip(shift_x, -3, 3)
        canvas_arr = shift(canvas_arr, shift=(shift_y, shift_x), mode="constant", cval=0.0)

    # 7. Normalize pixel intensity to [0.0, 1.0]
    normalized = canvas_arr / 255.0

    # 8. Flatten to (1, 784)
    return normalized.reshape(1, -1).astype(np.float32)


def array_to_image_preview(preprocessed_vector: np.ndarray) -> Image.Image:
    """
    Converts a preprocessed (1, 784) vector back into a 28x28 PIL preview image.
    """
    arr_2d = (preprocessed_vector.reshape(28, 28) * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr_2d)
