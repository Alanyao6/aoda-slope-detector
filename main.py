"""
main.py — entry point for the AI Image-Based Slope Detection Pipeline.

Usage:
    python main.py --image path/to/streetview_screenshot.jpg [--output results.json]

Pipeline:
    Input image
        ├─ Module 1: Monocular Depth Estimation  (depth.py)
        ├─ Module 2: Surface Normal Estimation   (normals.py)
        └─ Module 3: Semantic Segmentation       (segmentation.py)
                          ↓
             Module 4: Camera Pose Estimation    (camera_pose.py)
                          ↓
             Module 5: Scale Calibration         (scale_calibration.py)
                          ↓
             Module 6: 3D Point Cloud + Plane Fit (plane_fitting.py)
                          ↓
             Module 7: Slope Calculation + AODA  (slope_calc.py)
                          ↓
                       Output
"""

import argparse
import concurrent.futures
import os
import sys

from utils import load_image, save_json
from depth import estimate_depth
from normals import estimate_normals
from segmentation import segment_image
from camera_pose import estimate_camera_pose
from scale_calibration import calibrate_scale
from plane_fitting import reconstruct_and_fit
from slope_calc import calculate_slope


def run_pipeline(image_path: str, output_json: str | None = None) -> dict:
    """
    Execute the full slope detection pipeline.

    Args:
        image_path:  Path to the input image (.jpg or .png).
        output_json: Optional path to save a JSON report.

    Returns:
        Results dictionary.
    """
    all_warnings: list[str] = []

    # ------------------------------------------------------------------ #
    # Load image
    # ------------------------------------------------------------------ #
    print(f"Loading image: {image_path}")
    image_rgb = load_image(image_path)
    print(f"  Image shape: {image_rgb.shape}")

    # ------------------------------------------------------------------ #
    # Modules 1, 2, 3 — run in parallel using threads
    # (they don't depend on each other)
    # ------------------------------------------------------------------ #
    print("\nRunning Modules 1 (depth), 2 (normals), 3 (segmentation) in parallel …")

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        future_depth = executor.submit(estimate_depth, image_rgb)
        future_normals_placeholder = None  # normals need depth; see below
        future_seg = executor.submit(segment_image, image_rgb)

        depth_normalized, depth_raw = future_depth.result()
        print("  [Module 1] Depth estimation complete.")

        seg_result = future_seg.result()
        print("  [Module 3] Segmentation complete.")
        all_warnings.extend(seg_result["warnings"])

    # Module 2 depends on depth output — run after Module 1 finishes
    normal_map = estimate_normals(depth_normalized)
    print("  [Module 2] Normal estimation complete.")

    # ------------------------------------------------------------------ #
    # Module 4: Camera Pose Estimation
    # ------------------------------------------------------------------ #
    print("\n[Module 4] Estimating camera pose …")
    pose = estimate_camera_pose(image_rgb)
    print(f"  pitch = {pose['pitch']:.4f} rad  yaw = {pose['yaw']:.4f} rad")
    all_warnings.extend(pose["warnings"])

    # ------------------------------------------------------------------ #
    # Module 5: Scale Calibration
    # ------------------------------------------------------------------ #
    print("\n[Module 5] Calibrating depth scale …")
    scale_result = calibrate_scale(
        depth_raw=depth_raw,
        reference_object_mask=seg_result["reference_object_mask"],
        path_mask=seg_result["path_mask"],
    )
    print(f"  scale_factor = {scale_result['scale_factor']:.6f}  "
          f"(method: {scale_result['method']})")
    all_warnings.extend(scale_result["warnings"])

    # ------------------------------------------------------------------ #
    # Module 6: 3D Point Cloud + Plane Fitting
    # ------------------------------------------------------------------ #
    print("\n[Module 6] Reconstructing 3D point cloud and fitting plane …")
    fit_result = reconstruct_and_fit(
        absolute_depth_map=scale_result["absolute_depth_map"],
        path_mask=seg_result["path_mask"],
        R=pose["R"],
        f=pose["f"],
        cx=pose["cx"],
        cy=pose["cy"],
        scale_factor=scale_result["scale_factor"],
    )
    print(f"  plane normal = {fit_result['normal']}")
    print(f"  inlier ratio = {fit_result['inlier_ratio']:.3f}")
    all_warnings.extend(fit_result["warnings"])

    # ------------------------------------------------------------------ #
    # Module 7: Slope Calculation + AODA Check
    # ------------------------------------------------------------------ #
    print("\n[Module 7] Calculating slope and AODA compliance …")
    slope_result = calculate_slope(fit_result["normal"])

    # ------------------------------------------------------------------ #
    # Output
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 38)
    print("=== AI Slope Detection Results ===")
    print("=" * 38)
    print(f"Slope angle:          {slope_result['slope_degrees']:.2f} degrees")
    print(f"Slope grade:          {slope_result['slope_percent_grade']:.2f} %")
    print(f"AODA status:          {slope_result['aoda_status']}")
    print(f"Scale factor used:    {scale_result['scale_factor']:.6f}")
    print(f"Plane fit inlier ratio: {fit_result['inlier_ratio']:.3f}")
    if all_warnings:
        print("Warnings:")
        for w in all_warnings:
            print(f"  - {w}")
    else:
        print("Warnings: none")
    print("=" * 38)

    results = {
        "slope_degrees": round(slope_result["slope_degrees"], 4),
        "slope_percent_grade": round(slope_result["slope_percent_grade"], 4),
        "aoda_status": slope_result["aoda_status"],
        "scale_factor": scale_result["scale_factor"],
        "scale_method": scale_result["method"],
        "inlier_ratio": round(fit_result["inlier_ratio"], 4),
        "warnings": all_warnings,
        "input_image": os.path.abspath(image_path),
    }

    if output_json:
        save_json(results, output_json)
        print(f"\nResults saved to: {output_json}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AI Image-Based Slope Detection Pipeline"
    )
    parser.add_argument(
        "image",
        help="Path to the input image (.jpg or .png)."
    )
    parser.add_argument(
        "--output", default=None,
        help="Optional path to save a JSON report. Defaults to <image_name>_results.json."
    )
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        print(f"Error: image file not found: {args.image}", file=sys.stderr)
        sys.exit(1)

    if args.output is None:
        base = os.path.splitext(args.image)[0]
        args.output = f"{base}_results.json"

    run_pipeline(args.image, output_json=args.output)


if __name__ == "__main__":
    main()
