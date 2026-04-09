"""Module 1: Monocular Depth Estimation using Depth Anything V2."""

import numpy as np
import torch
from PIL import Image
from transformers import pipeline as hf_pipeline


MODEL_ID = "depth-anything/Depth-Anything-V2-Large-hf"


def estimate_depth(image_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Run Depth Anything V2 on the input image.

    Args:
        image_rgb: RGB image as numpy array (H x W x 3), uint8.

    Returns:
        depth_normalized: Depth map normalized to [0, 1], float32, shape (H, W).
                          Higher values = farther from camera.
        depth_raw:        Unnormalized raw depth map, float32, shape (H, W).
                          Used by Module 5 for scale calibration.
    """
    H, W = image_rgb.shape[:2]

    device = 0 if torch.cuda.is_available() else -1
    pipe = hf_pipeline(
        task="depth-estimation",
        model=MODEL_ID,
        device=device,
    )

    pil_img = Image.fromarray(image_rgb)
    result = pipe(pil_img)

    # The pipeline returns a PIL Image in result["depth"]
    depth_pil: Image.Image = result["depth"]

    # Resize back to original resolution using bilinear interpolation
    depth_pil = depth_pil.resize((W, H), Image.BILINEAR)
    depth_raw = np.array(depth_pil, dtype=np.float32)

    # Normalize to [0, 1]
    d_min, d_max = depth_raw.min(), depth_raw.max()
    if d_max - d_min < 1e-6:
        depth_normalized = np.zeros_like(depth_raw)
    else:
        depth_normalized = (depth_raw - d_min) / (d_max - d_min)

    return depth_normalized, depth_raw
