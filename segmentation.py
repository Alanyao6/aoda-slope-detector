"""Module 3: Semantic Segmentation using Mask2Former (ADE20K).

Produces:
  - path_mask          : pixels belonging to pavement / sidewalk / road
  - vegetation_mask    : pixels belonging to grass / earth / field
  - reference_object_mask : pixels belonging to known-size objects (cars, bins, signs)
"""

import warnings
import numpy as np
import cv2
from PIL import Image
import torch
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation


MODEL_ID = "facebook/mask2former-swin-large-ade-semantic"

# ADE20K class names that map to each category
# (checked against the 150-class ADE20K label list)
_PATH_KEYWORDS = {
    "road", "sidewalk", "path", "runway", "dirt track",
    "floor", "pavement", "ground",
}
_VEGETATION_KEYWORDS = {
    "grass", "earth", "field", "land", "sand", "soil",
    "vegetation", "plant", "tree", "shrub", "bush",
}
_REFERENCE_KEYWORDS = {
    "car", "van", "truck", "bus", "minibike", "motorcycle",
    "traffic sign", "sign", "signboard",
    "trash can", "ashcan", "dustbin", "garbage can", "waste bin",
    "barrel", "bin",
}


def _keyword_match(label: str, keywords: set) -> bool:
    label_lower = label.lower()
    return any(kw in label_lower for kw in keywords)


def _morphological_close(mask: np.ndarray, ksize: int = 5) -> np.ndarray:
    kernel = np.ones((ksize, ksize), dtype=np.uint8)
    return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)


def segment_image(image_rgb: np.ndarray) -> dict:
    """
    Run semantic segmentation on the input image.

    Args:
        image_rgb: RGB image as numpy array (H x W x 3), uint8.

    Returns:
        dict with keys:
          "seg_map"               : (H, W) int array of class IDs
          "id2label"              : dict mapping int class ID -> str label
          "path_mask"             : (H, W) bool
          "vegetation_mask"       : (H, W) bool
          "reference_object_mask" : (H, W) bool
          "warnings"              : list of str
    """
    H, W = image_rgb.shape[:2]
    seg_warnings: list[str] = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(MODEL_ID)
    model.to(device)
    model.eval()

    pil_img = Image.fromarray(image_rgb)
    inputs = processor(images=pil_img, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    # Post-process: get per-pixel class predictions
    result = processor.post_process_semantic_segmentation(
        outputs, target_sizes=[(H, W)]
    )[0]  # shape (H, W), values are class IDs

    seg_map = result.cpu().numpy().astype(np.int32)
    id2label: dict[int, str] = model.config.id2label  # type: ignore[attr-defined]

    # Build category masks
    path_mask = np.zeros((H, W), dtype=bool)
    vegetation_mask = np.zeros((H, W), dtype=bool)
    reference_object_mask = np.zeros((H, W), dtype=bool)

    for class_id, label in id2label.items():
        pixels = seg_map == int(class_id)
        if not pixels.any():
            continue
        if _keyword_match(label, _PATH_KEYWORDS):
            path_mask |= pixels
        if _keyword_match(label, _VEGETATION_KEYWORDS):
            vegetation_mask |= pixels
        if _keyword_match(label, _REFERENCE_KEYWORDS):
            reference_object_mask |= pixels

    # Morphological closing to fill small holes
    path_mask = _morphological_close(path_mask, ksize=5)
    vegetation_mask = _morphological_close(vegetation_mask, ksize=5)
    reference_object_mask = _morphological_close(reference_object_mask, ksize=5)

    # Fallback: if no path detected, use largest connected region in lower half
    if not path_mask.any():
        seg_warnings.append(
            "No path/sidewalk/road detected by segmentation. "
            "Falling back to largest connected region in lower half of image."
        )
        lower_half = np.zeros((H, W), dtype=np.uint8)
        lower_half[H // 2:, :] = 1

        # Use the most common non-sky class in the lower half as the path
        lower_seg = seg_map * lower_half
        unique, counts = np.unique(lower_seg, return_counts=True)
        # Exclude 0 if it maps to sky/background
        best_class = None
        best_count = 0
        for uid, cnt in zip(unique, counts):
            label = id2label.get(int(uid), "")
            if "sky" in label.lower() or uid == 0:
                continue
            if cnt > best_count:
                best_count = cnt
                best_class = uid

        if best_class is not None:
            # Find connected components of that class in lower half
            candidate = ((seg_map == best_class) & lower_half.astype(bool)).astype(np.uint8)
            num_labels, labels_cc, stats, _ = cv2.connectedComponentsWithStats(candidate)
            if num_labels > 1:
                largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                path_mask = labels_cc == largest
            else:
                path_mask = candidate.astype(bool)
        else:
            # Last resort: bottom 20% of image
            path_mask[int(0.8 * H):, :] = True
            seg_warnings.append("Could not identify any surface class; using bottom 20% of image as path.")

    return {
        "seg_map": seg_map,
        "id2label": id2label,
        "path_mask": path_mask,
        "vegetation_mask": vegetation_mask,
        "reference_object_mask": reference_object_mask,
        "warnings": seg_warnings,
    }
