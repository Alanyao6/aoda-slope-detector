"""Module 5: Scale Calibration.

Converts the relative depth map into absolute depth (meters) using either:
  1. A known-size reference object (Toronto waste bin, height = 1.05 m).
  2. Fallback: Google Street View camera height (~2.5 m above ground).
"""

import numpy as np


WASTE_BIN_HEIGHT_M = 1.05   # Toronto standard waste bin height in metres
STREET_VIEW_CAM_HEIGHT_M = 2.5  # Approximate Street View camera mounting height


def calibrate_scale(
    depth_raw: np.ndarray,
    reference_object_mask: np.ndarray,
    path_mask: np.ndarray,
) -> dict:
    """
    Estimate a scale factor to convert relative depth to metres.

    Args:
        depth_raw:               Raw (unnormalized) depth map, float32, (H, W).
        reference_object_mask:   Boolean mask of reference objects, (H, W).
        path_mask:               Boolean mask of path/pavement, (H, W).

    Returns:
        dict with keys:
          "absolute_depth_map" : (H, W) float32, depth in metres
          "scale_factor"       : float
          "method"             : str, which calibration method was used
          "warnings"           : list of str
    """
    H, W = depth_raw.shape
    scale_warnings: list[str] = []
    scale_factor: float = 1.0
    method = "none"

    # --- Method 1: waste bin bounding box ---
    if reference_object_mask.any():
        rows = np.where(reference_object_mask)[0]
        row_top = rows.min()
        row_bot = rows.max()
        if row_bot > row_top:
            # Sample a horizontal strip at top and bottom of the bounding box
            cols = np.where(reference_object_mask)[1]
            col_l = cols.min()
            col_r = cols.max()

            strip_h = max(1, (row_bot - row_top) // 10)
            depth_top = float(np.median(
                depth_raw[row_top: row_top + strip_h, col_l: col_r + 1]
            ))
            depth_bot = float(np.median(
                depth_raw[row_bot - strip_h: row_bot + 1, col_l: col_r + 1]
            ))
            depth_diff = abs(depth_bot - depth_top)

            if depth_diff > 1e-6:
                scale_factor = WASTE_BIN_HEIGHT_M / depth_diff
                method = "waste_bin"
            else:
                scale_warnings.append(
                    "Reference object detected but depth difference is near zero; "
                    "skipping reference-object calibration."
                )

    # --- Method 2: Street View camera height fallback ---
    if method == "none":
        if path_mask.any():
            # Bottom 10% of rows in the path_mask region
            bottom_row_start = int(0.9 * H)
            ground_region = path_mask[bottom_row_start:, :]
            ground_depth_values = depth_raw[bottom_row_start:, :][ground_region]

            if ground_depth_values.size > 0:
                ground_depth_median = float(np.median(ground_depth_values))
                if ground_depth_median > 1e-6:
                    scale_factor = STREET_VIEW_CAM_HEIGHT_M / ground_depth_median
                    method = "camera_height"
                else:
                    scale_warnings.append(
                        "Ground depth median is near zero; cannot use camera-height calibration."
                    )
            else:
                scale_warnings.append(
                    "No path pixels in bottom 10% of image for camera-height calibration."
                )
        else:
            scale_warnings.append("path_mask is empty; cannot use camera-height calibration.")

    if method == "none":
        scale_warnings.append(
            "Both calibration methods failed. scale_factor = 1.0. "
            "Output depth is in relative (arbitrary) units."
        )

    absolute_depth_map = (depth_raw * scale_factor).astype(np.float32)

    return {
        "absolute_depth_map": absolute_depth_map,
        "scale_factor": scale_factor,
        "method": method,
        "warnings": scale_warnings,
    }
