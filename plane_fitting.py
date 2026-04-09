"""Module 6: 3D Point Cloud Reconstruction and RANSAC Plane Fitting.

Back-projects image pixels to 3D using the absolute depth map, applies camera
pose correction, then fits a plane to the path-surface points via RANSAC.
"""

import numpy as np


MAX_POINTS = 50_000
RANSAC_ITERATIONS = 1000
MIN_INLIER_RATIO_WARN = 0.3


def _plane_from_3_points(pts: np.ndarray) -> np.ndarray | None:
    """
    Compute plane equation (a, b, c, d) from 3 points.
    Returns unit normal (a,b,c) and offset d, or None if degenerate.
    """
    p0, p1, p2 = pts[0], pts[1], pts[2]
    v1 = p1 - p0
    v2 = p2 - p0
    normal = np.cross(v1, v2)
    norm = np.linalg.norm(normal)
    if norm < 1e-10:
        return None
    normal = normal / norm
    d = -np.dot(normal, p0)
    return np.array([normal[0], normal[1], normal[2], d])


def _point_plane_distance(points: np.ndarray, plane: np.ndarray) -> np.ndarray:
    """Signed distance of each point from the plane (a,b,c,d)."""
    abc = plane[:3]
    d = plane[3]
    return np.abs(points @ abc + d)  # |ax+by+cz+d| / ||(a,b,c)|| = 1 (normalized)


def ransac_plane_fit(
    points: np.ndarray,
    distance_threshold: float,
) -> dict:
    """
    RANSAC plane fitting.

    Args:
        points:             (N, 3) float array of 3-D points.
        distance_threshold: Inlier distance tolerance in metres.

    Returns:
        dict with keys:
          "plane"         : (4,) array [a, b, c, d] (unit normal + offset)
          "normal"        : (3,) unit normal vector
          "inlier_ratio"  : float
          "inlier_mask"   : (N,) bool
    """
    N = len(points)
    best_plane = None
    best_inlier_count = 0
    best_inlier_mask = np.zeros(N, dtype=bool)

    rng = np.random.default_rng(42)

    for _ in range(RANSAC_ITERATIONS):
        idx = rng.choice(N, size=3, replace=False)
        plane = _plane_from_3_points(points[idx])
        if plane is None:
            continue
        dists = _point_plane_distance(points, plane)
        inliers = dists < distance_threshold
        count = int(inliers.sum())
        if count > best_inlier_count:
            best_inlier_count = count
            best_plane = plane
            best_inlier_mask = inliers

    # Refit using all inliers via least-squares
    if best_plane is not None and best_inlier_mask.sum() >= 3:
        inlier_pts = points[best_inlier_mask]
        centroid = inlier_pts.mean(axis=0)
        centered = inlier_pts - centroid
        _, _, Vt = np.linalg.svd(centered, full_matrices=False)
        normal = Vt[-1]  # eigenvector with smallest singular value
        normal = normal / np.linalg.norm(normal)
        d = -np.dot(normal, centroid)
        best_plane = np.array([normal[0], normal[1], normal[2], d])
        dists = _point_plane_distance(points, best_plane)
        best_inlier_mask = dists < distance_threshold

    inlier_ratio = best_inlier_mask.sum() / N if N > 0 else 0.0

    return {
        "plane": best_plane,
        "normal": best_plane[:3] if best_plane is not None else np.array([0.0, 1.0, 0.0]),
        "inlier_ratio": inlier_ratio,
        "inlier_mask": best_inlier_mask,
    }


def reconstruct_and_fit(
    absolute_depth_map: np.ndarray,
    path_mask: np.ndarray,
    R: np.ndarray,
    f: float,
    cx: float,
    cy: float,
    scale_factor: float,
) -> dict:
    """
    Build a 3-D point cloud, apply camera pose correction, extract path points,
    and run RANSAC plane fitting.

    Args:
        absolute_depth_map: (H, W) float32, depth in metres.
        path_mask:          (H, W) bool.
        R:                  (3, 3) rotation matrix (camera → world).
        f, cx, cy:          Camera intrinsics (pixels).
        scale_factor:       Used to set the RANSAC distance threshold.

    Returns:
        dict with keys:
          "normal"       : (3,) plane normal in world coords
          "plane"        : (4,) plane equation [a,b,c,d]
          "inlier_ratio" : float
          "warnings"     : list of str
    """
    H, W = absolute_depth_map.shape
    fit_warnings: list[str] = []

    # Back-project all pixels
    u = np.arange(W, dtype=np.float32)
    v = np.arange(H, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)  # (H, W)

    d = absolute_depth_map  # (H, W)
    X = (uu - cx) * d / f
    Y = (vv - cy) * d / f
    Z = d

    # Stack into (H*W, 3)
    pts_cam = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)

    # Apply camera pose rotation: world = R @ camera
    pts_world = (R @ pts_cam.T).T  # (N, 3)

    # Extract path points
    path_flat = path_mask.reshape(-1)
    path_pts = pts_world[path_flat]

    # Subsample if too many points
    if len(path_pts) > MAX_POINTS:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(path_pts), size=MAX_POINTS, replace=False)
        path_pts = path_pts[idx]

    if len(path_pts) < 3:
        fit_warnings.append("Not enough path points to fit a plane.")
        return {
            "normal": np.array([0.0, 1.0, 0.0]),
            "plane": np.array([0.0, 1.0, 0.0, 0.0]),
            "inlier_ratio": 0.0,
            "warnings": fit_warnings,
        }

    distance_threshold = 0.05 * max(scale_factor, 1.0)
    result = ransac_plane_fit(path_pts, distance_threshold=distance_threshold)

    if result["inlier_ratio"] < MIN_INLIER_RATIO_WARN:
        fit_warnings.append(
            f"Low plane fit inlier ratio ({result['inlier_ratio']:.2f}). "
            "Path surface may be irregular or segmentation may be poor."
        )

    return {
        "normal": result["normal"],
        "plane": result["plane"],
        "inlier_ratio": result["inlier_ratio"],
        "warnings": fit_warnings,
    }
