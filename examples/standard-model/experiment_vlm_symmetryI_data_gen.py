"""
Generate 20 MNIST-like images with digit color transitioning from black to red,
on a light gray background. Images are saved in the test_data folder as PNG.
"""

import os
import numpy as np
from PIL import Image
from sklearn.datasets import load_digits

# Load sklearn digits dataset (8x8 images)
digits = load_digits()

# Pick one sample digit (e.g., digit "5")
target_digit = 5
indices = np.where(digits.target == target_digit)[0]
sample = digits.images[indices[0]]  # shape (8, 8), values 0-16

# Upscale factor to make images larger (8x8 -> 64x64)
SCALE = 8
IMG_SIZE = 8 * SCALE  # 64

# Number of images to generate
N = 20

# Output folder
output_dir = os.path.join(os.path.dirname(__file__), "test_data")
os.makedirs(output_dir, exist_ok=True)

# Light gray background color
BG_COLOR = (211, 211, 211)  # RGB light gray

for i in range(N):
    # Interpolation factor: 0.0 = black, 1.0 = red
    t = i / (N - 1)

    # Digit color: interpolate from black (0,0,0) to red (255,0,0)
    digit_color = (int(255 * t), 0, 0)

    # Normalize the 8x8 digit image to [0, 1] (0 = background, 1 = foreground)
    digit_norm = sample / sample.max()

    # Create RGB image filled with background color
    rgb = np.full((8, 8, 3), BG_COLOR, dtype=np.float32)

    # Blend each pixel: where digit_norm=1 -> digit_color, where digit_norm=0 -> BG_COLOR
    for c in range(3):
        rgb[:, :, c] = (
            (1 - digit_norm) * BG_COLOR[c] + digit_norm * digit_color[c]
        )

    rgb = rgb.astype(np.uint8)

    # Upscale using nearest-neighbor to preserve sharp pixel look
    img = Image.fromarray(rgb, mode="RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE), resample=Image.NEAREST)

    filename = os.path.join(output_dir, f"mnist_digit_{i + 1:02d}.png")
    img.save(filename)

