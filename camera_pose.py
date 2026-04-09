"""Module 4: Camera Pose Estimation via Vanishing Point Detection.

Uses classical CV (Canny + Hough lines) to estimate camera pitch and yaw.
Returns a 3x3 rotation matrix R = Ry(yaw) @ Rx(pitch) used to rotate
camera-space points into world space.
"""

import math
import warnings

import cv2
import numpy as np


def _line_to_params(x1, y1, x2, y2):
    """Return (a, b, c) coefficients of line ax + by + c = 0."""
    a = y2 - y1
    b = x1 - x2
    c = x2 * y1 - x1 * y2
    return a, b, c


def _least_squares_intersection(lines_abc: list[tuple]) -> tuple[float, float] | None:
    """
    Find the least-squares intersection of multiple lines given as (a, b, c).
    Solves the over-determined system [a, b] [x, y]^T = -c in the LS sense.
    Returns (x, y) or None if the system is degenerate.
    """
    A_rows = []
    b_rows = []
    for a, b, c in lines_abc:
        A_rows.append([a, b])
        b_rows.append([-c])
    A = np.array(A_rows, dtype=np.float64)
    b = np.array(b_rows, dtype=np.float64)
    result, residuals, rank, _ = np.linalg.lstsq(A, b, rcond=None)
    if rank < 2:
        return None
    return float(result[0]), float(result[1])


def _rotation_x(angle: float) -> np.ndarray:
    """Rotation matrix around X axis."""
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s,  c]], dtype=np.float64)


def _rotation_y(angle: float) -> np.ndarray:
    """Rotation matrix around Y axis."""
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[ c, 0, s],
                     [ 0, 1, 0],
                     [-s, 0, c]], dtype=np.float64)


def estimate_camera_pose(image_rgb: np.ndarray) -> dict:
    """
    Estimate camera pitch and yaw from vanishing points.

    Args:
        image_rgb: RGB image, uint8, shape (H, W, 3).

    Returns:
        dict with keys:
          "pitch"  : float, radians
          "yaw"    : float, radians
          "R"      : (3, 3) float64 rotation matrix  R = Ry(yaw) @ Rx(pitch)
          "f"      : float, estimated focal length in pixels
          "cx"     : float, principal point x
          "cy"     : float, principal point y
          "warnings": list of str
    """
    H, W = image_rgb.shape[:2]
    cx, cy = W / 2.0, H / 2.0
    f = float(W)  # focal length assumption: f = image width
    pose_warnings: list[str] = []

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=math.pi / 180,
        threshold=80,
        minLineLength=60,
        maxLineGap=10,
    )

    vertical_lines_abc: list[tuple] = []
    horizontal_lines_abc: list[tuple] = []

    if lines is not None:
        for seg in lines:
            x1, y1, x2, y2 = seg[0]
            dx = x2 - x1
            dy = y2 - y1
            length = math.hypot(dx, dy)
            if length < 1e-6:
                continue
            angle_from_vertical = math.degrees(math.atan2(abs(dx), abs(dy)))
            angle_from_horizontal = 90.0 - angle_from_vertical
            abc = _line_to_params(x1, y1, x2, y2)
            if angle_from_vertical <= 30.0:
                vertical_lines_abc.append(abc)
            elif angle_from_horizontal <= 30.0 and angle_from_horizontal > 1.0:
                # exclude truly horizontal lines for yaw (they don't converge usefully)
                horizontal_lines_abc.append(abc)

    pitch = 0.0
    yaw = 0.0

    # --- Vertical vanishing point → pitch ---
    if len(vertical_lines_abc) >= 4:
        vp = _least_squares_intersection(vertical_lines_abc)
        if vp is not None:
            vpy_y = vp[1]
            pitch = math.atan2(vpy_y - cy, f)
    else:
        pose_warnings.append(
            f"Only {len(vertical_lines_abc)} near-vertical lines detected "
            f"(need ≥4). Assuming pitch = 0."
        )

    # --- Horizontal vanishing point → yaw ---
    if len(horizontal_lines_abc) >= 4:
        vp_h = _least_squares_intersection(horizontal_lines_abc)
        if vp_h is not None:
            vpx_x = vp_h[0]
            yaw = math.atan2(vpx_x - cx, f)
    else:
        pose_warnings.append(
            f"Only {len(horizontal_lines_abc)} near-horizontal lines detected "
            f"(need ≥4). Assuming yaw = 0."
        )

    R = _rotation_y(yaw) @ _rotation_x(pitch)

    return {
        "pitch": pitch,
        "yaw": yaw,
        "R": R,
        "f": f,
        "cx": cx,
        "cy": cy,
        "warnings": pose_warnings,
    }
