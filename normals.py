"""Module 2: Surface Normal Estimation derived from the depth map.

Uses depth-to-normal conversion (Sobel gradient method) as the primary approach,
since it requires no additional model download.

Coordinate convention: x = right, y = down, z = into the screen.
"""

import numpy as np
from scipy.ndimage import gaussian_filter


def estimate_normals(depth_normalized: np.ndarray) -> np.ndarray:
    """
    Derive per-pixel surface normals from a normalized depth map.

    Steps:
      1. Smooth depth with a Gaussian (sigma=1) to reduce noise.
      2. Compute partial derivatives dD/dx and dD/dy with Sobel filters.
      3. Normal = normalize((-dD/dx, -dD/dy, 1)).

    Args:
        depth_normalized: Normalized depth map, float32, shape (H, W), range [0, 1].

    Returns:
        normal_map: Shape (H, W, 3), each pixel is a unit vector (nx, ny, nz).
                    Convention: x=right, y=down, z=into-screen.
    """
    depth_smooth = gaussian_filter(depth_normalized.astype(np.float64), sigma=1)

    # Sobel kernels
    sobel_x = np.array([[-1, 0, 1],
                         [-2, 0, 2],
                         [-1, 0, 1]], dtype=np.float64)
    sobel_y = np.array([[-1, -2, -1],
                         [ 0,  0,  0],
                         [ 1,  2,  1]], dtype=np.float64)

    from scipy.signal import convolve2d
    dDdx = convolve2d(depth_smooth, sobel_x, mode="same", boundary="symm")
    dDdy = convolve2d(depth_smooth, sobel_y, mode="same", boundary="symm")

    # Normal components: n = (-dD/dx, -dD/dy, 1), then normalize
    nx = -dDdx
    ny = -dDdy
    nz = np.ones_like(nx)

    norm = np.sqrt(nx**2 + ny**2 + nz**2)
    norm = np.where(norm < 1e-8, 1.0, norm)

    normal_map = np.stack([nx / norm, ny / norm, nz / norm], axis=-1).astype(np.float32)
    return normal_map
