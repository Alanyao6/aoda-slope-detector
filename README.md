# AI Image-Based Slope Detector

Estimates the slope angle of a walkway or road surface from a single street-view photograph and checks whether it meets the **AODA** (Accessibility for Ontarians with Disabilities Act) accessibility standard.

---

## How It Works

The pipeline runs 7 modules in sequence:

```
Input Image
  ├─ Module 1: Monocular Depth Estimation   (Depth Anything V2)
  ├─ Module 2: Surface Normal Estimation    (from depth map)
  └─ Module 3: Semantic Segmentation        (Mask2Former / ADE20K)
                    ↓
       Module 4: Camera Pose Estimation     (vanishing point detection)
                    ↓
       Module 5: Scale Calibration          (reference object or camera height)
                    ↓
       Module 6: 3D Point Cloud + Plane Fit (RANSAC)
                    ↓
       Module 7: Slope Calculation + AODA Check
```

| Module | File | What it does |
|--------|------|-------------|
| 1 | `depth.py` | Runs **Depth Anything V2 Large** to produce a relative depth map |
| 2 | `normals.py` | Derives surface normals from the depth map using finite differences |
| 3 | `segmentation.py` | Runs **Mask2Former** (ADE20K) to identify road/path pixels and reference objects |
| 4 | `camera_pose.py` | Detects vanishing points via Hough lines to estimate camera pitch and yaw |
| 5 | `scale_calibration.py` | Converts relative depth to metres using a waste bin (1.05 m) or Street View camera height (2.5 m) |
| 6 | `plane_fitting.py` | Builds a 3D point cloud from the path mask and fits a plane with RANSAC |
| 7 | `slope_calc.py` | Computes slope angle from the plane normal and checks AODA thresholds |

### AODA Thresholds

| Status | Slope Grade |
|--------|------------|
| PASS | ≤ 5% |
| CONDITIONAL | 5% – 8.33% |
| FAIL | > 8.33% |

---

## Requirements

- Python 3.10+
- PyTorch (CPU or CUDA)
- Transformers (`pip install transformers`)
- OpenCV (`pip install opencv-python`)
- NumPy, Pillow, Flask

Install all dependencies:

```bash
pip install torch torchvision transformers opencv-python numpy pillow flask
```

> **First run** will download ~1.5 GB of model weights (Depth Anything V2 Large + Mask2Former). Weights are cached locally by HuggingFace after the first download.

---

## Usage

### Command Line

```bash
cd slope_detector
python main.py path/to/image.jpg
```

Results are printed to the terminal and saved to `<image_name>_results.json`.

Optional output path:

```bash
python main.py path/to/image.jpg --output my_results.json
```

### Web App

```bash
cd slope_detector
python app.py
```

Open `http://127.0.0.1:5000` in your browser. Drag and drop a JPG or PNG (max 16 MB) and click **Analyze Slope**.

---

## Output

```json
{
  "slope_degrees": 7.59,
  "slope_percent_grade": 13.33,
  "aoda_status": "FAIL",
  "scale_factor": 0.038889,
  "scale_method": "waste_bin",
  "inlier_ratio": 0.997,
  "warnings": [],
  "input_image": "/absolute/path/to/image.jpg"
}
```

| Field | Description |
|-------|-------------|
| `slope_degrees` | Estimated slope angle in degrees |
| `slope_percent_grade` | Slope as a percentage (rise/run × 100) |
| `aoda_status` | `PASS`, `CONDITIONAL`, or `FAIL` |
| `scale_factor` | Depth-to-metres conversion factor used |
| `scale_method` | `waste_bin`, `camera_height`, or `none` (relative units) |
| `inlier_ratio` | RANSAC plane fit quality (0–1; higher is better) |
| `warnings` | List of degraded-quality alerts from any module |

---

## Accuracy Notes

Results are most reliable when the input image has:

- **Clear vertical structures** (building edges, door frames, poles) — needed for camera pitch estimation. If fewer than 4 vertical lines are detected, pitch is assumed to be 0, which can add several degrees of error.
- **A recognizable reference object** (waste bin, car, sign) — used for real-world scale calibration. Without one, the pipeline falls back to camera-height estimation; if that also fails, `scale_factor = 1.0` and depth is in relative units.
- **A resolution close to the model's training size** — very large images should be resized before input to reduce noise in depth and segmentation outputs.

A low `inlier_ratio` (< 0.4) or a `scale_method` of `none` in the output indicates the result should be treated with caution.
