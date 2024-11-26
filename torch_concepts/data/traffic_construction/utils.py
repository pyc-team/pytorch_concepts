import cv2
import numpy as np
import os

from PIL import Image
from scipy.ndimage import rotate

def resize_with_aspect_ratio(img, target_width=None, target_height=None):
    # Convert the NumPy array to a PIL Image
    img_uint8 = (img * 255).astype(np.uint8)
    pil_img = Image.fromarray(img_uint8)

    # Get original dimensions
    original_width, original_height = pil_img.size

    # Calculate the new dimensions while maintaining aspect ratio
    if target_width is not None:
        # Calculate height based on target width
        assert target_height is None
        if target_width <= 1:
            target_width = int(np.ceil(original_width * target_width))
        scale_ratio = target_width / original_width
        new_width = target_width
        new_height = int(original_height * scale_ratio)
    elif target_height is not None:
        # Calculate width based on target height
        if target_height <= 1:
            target_height = int(np.ceil(original_height * target_height))
        scale_ratio = target_height / original_height
        new_height = target_height
        new_width = int(original_width * scale_ratio)
    else:
        # No target specified, return original image
        return img

    # Resize the image
    resized_img = pil_img.resize((new_width, new_height), Image.ANTIALIAS)

    # Convert back to NumPy array and normalize to [0, 1]
    resized_img = np.array(resized_img) / 255.0

    return resized_img

def add_sprite(
    sprite,
    background,
    target_width=None,
    target_height=None,
    rotation=0,
    x_offset=None,
    y_offset=None,
    inplace=False,
):
    sprite = resize_with_aspect_ratio(
        sprite,
        target_width=target_width,
        target_height=target_height,
    )
    sprite = rotate(sprite, angle=rotation, reshape=True)
    if inplace:
        canvas = background[:, :, :3]
    else:
        canvas = np.zeros((background.shape[0], background.shape[1], 3))
        canvas[:] = background[:, :, :3]

    # Calculate the position to center img2 on the canvas
    x_offset = (
        (canvas.shape[1] - sprite.shape[1]) // 2
        if x_offset is None else x_offset
    )
    y_offset = (
        (canvas.shape[0] - sprite.shape[0]) // 2
        if y_offset is None else y_offset
    )

    # Place sprite onto the center of the canvas
    alpha = sprite[:, :, 3:]
    x_sprite_start = -x_offset if x_offset < 0 else 0
    if canvas.shape[1] < sprite.shape[1] + x_offset:
        x_sprite_end = max(
            sprite.shape[1] - (sprite.shape[1] + x_offset - canvas.shape[1]),
            0,
        )
    else:
        x_sprite_end = sprite.shape[1]
    eff_width = x_sprite_end - x_sprite_start

    y_sprite_start = -y_offset if y_offset < 0 else 0
    if canvas.shape[0] < sprite.shape[0] + y_offset:
        y_sprite_end = max(
            sprite.shape[0] - (sprite.shape[0] + y_offset - canvas.shape[0]),
            0,
        )
    else:
        y_sprite_end = sprite.shape[0]
    eff_height = y_sprite_end - y_sprite_start
    x_offset = max(x_offset, 0)
    y_offset = max(y_offset, 0)
    if (eff_height > 0) and (eff_width > 0):
        canvas[
            y_offset:(eff_height + y_offset),
            x_offset:(eff_width + x_offset),
            :
        ] = (
            sprite[y_sprite_start:y_sprite_end, x_sprite_start:x_sprite_end, :3] * alpha[y_sprite_start:y_sprite_end, x_sprite_start:x_sprite_end, :] +
            canvas[y_offset:(eff_height + y_offset), x_offset:(eff_width + x_offset), :] * (1 - alpha[y_sprite_start:y_sprite_end, x_sprite_start:x_sprite_end, :3])
        )
    if inplace:
        canvas[:] = np.clip(canvas, 0, 1)
        return canvas
    return np.clip(canvas, 0, 1)



def transform_scale_coordinates(x, y, ratio):
    new_x = round(x * ratio)
    new_y = round(y * ratio)
    return new_x, new_y



def highlight_edges(img, thickness=5):
    # Convert the image to [0, 255] range for OpenCV compatibility
    img_uint8 = (img[..., :3] * img[..., 3:] * 255).astype(np.uint8)

    # Convert to grayscale
    grayscale = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)

    # Apply Canny edge detection
    edges = cv2.Canny(grayscale, threshold1=100, threshold2=200)

    # Dilate the edges to make them thicker
    if thickness > 1:
        edges = cv2.dilate(edges, np.ones((thickness, thickness), np.uint8))

    red_mask = np.zeros_like(img)
    red_mask[..., 0] = edges / 255
    red_mask[..., 1] = 0
    red_mask[..., 2] = 0
    red_mask[..., 3] = edges / 255

    # Overlay the yellow edge mask on the original image
    highlighted_img = np.clip(img + red_mask * 0.8, 0, 1)

    return highlighted_img



def get_directory_size(directory):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            # Only add file sizes (skip if there's an error reading the file)
            try:
                total_size += os.path.getsize(file_path)
            except FileNotFoundError:
                # Skip files that can't be accessed (e.g., broken symlinks)
                pass
    return total_size

def format_size(size_in_bytes):
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_in_bytes < 1024:
            return f"{size_in_bytes:.2f} {unit}"
        size_in_bytes /= 1024