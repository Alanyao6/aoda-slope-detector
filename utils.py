"""Shared utilities: image loading and common helpers."""

import numpy as np
from PIL import Image


def load_image(path: str) -> np.ndarray:
    """Load an image from disk and return as RGB numpy array (H x W x 3), uint8."""
    img = Image.open(path).convert("RGB")
    return np.array(img)


def save_json(data: dict, path: str) -> None:
    """Save a dictionary as a JSON file."""
    import json
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
