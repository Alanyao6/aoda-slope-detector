"""Module 7: Slope Calculation and AODA Compliance Check."""

import math
import numpy as np


# AODA thresholds (percent grade)
AODA_PASS_MAX = 5.0
AODA_CONDITIONAL_MAX = 8.33


def calculate_slope(normal: np.ndarray) -> dict:
    """
    Compute slope angle from the fitted plane normal and check AODA compliance.

    Convention: after camera-pose correction, Y is the gravity/vertical axis
    (pointing up in world space). A perfectly flat surface has normal = (0,1,0).

    Args:
        normal: (3,) unit normal vector of the fitted plane in world coords.

    Returns:
        dict with keys:
          "slope_degrees"      : float
          "slope_percent_grade": float
          "aoda_status"        : str  ("PASS", "CONDITIONAL", or "FAIL")
    """
    vertical = np.array([0.0, 1.0, 0.0])

    n = np.array(normal, dtype=np.float64)
    n_norm = np.linalg.norm(n)
    if n_norm < 1e-8:
        # Degenerate normal – assume flat
        return {
            "slope_degrees": 0.0,
            "slope_percent_grade": 0.0,
            "aoda_status": "PASS",
        }

    n = n / n_norm
    cos_theta = abs(float(np.dot(n, vertical)))
    cos_theta = min(1.0, cos_theta)  # numerical safety
    theta_rad = math.acos(cos_theta)

    slope_degrees = math.degrees(theta_rad)
    slope_percent_grade = math.tan(theta_rad) * 100.0

    if slope_percent_grade <= AODA_PASS_MAX:
        aoda_status = "PASS"
    elif slope_percent_grade <= AODA_CONDITIONAL_MAX:
        aoda_status = "CONDITIONAL"
    else:
        aoda_status = "FAIL"

    return {
        "slope_degrees": slope_degrees,
        "slope_percent_grade": slope_percent_grade,
        "aoda_status": aoda_status,
    }
