"""
Flask web app for the AI Slope Detection Pipeline.

Usage:
    python app.py
    Then open http://127.0.0.1:5000 in your browser.
"""

import os
import uuid

from flask import Flask, request, render_template_string, url_for

# Import the existing pipeline function
from main import run_pipeline

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB max upload

# ---------------------------------------------------------------------------
# HTML Templates
# ---------------------------------------------------------------------------

INDEX_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Slope Detector</title>
    <style>
        :root {
            --blue: #3b82f6;
            --blue-dark: #2563eb;
            --green: #22c55e;
            --bg: #f8fafc;
            --card: #ffffff;
            --border: #e2e8f0;
            --text: #1e293b;
            --muted: #94a3b8;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: var(--bg);
            color: var(--text);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 24px;
        }
        header {
            text-align: center;
            margin-bottom: 32px;
        }
        header h1 {
            font-size: 2rem;
            font-weight: 700;
            letter-spacing: -0.5px;
        }
        header p {
            color: var(--muted);
            margin-top: 6px;
            font-size: 0.95rem;
        }
        .card {
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 16px;
            box-shadow: 0 4px 24px rgba(0,0,0,0.06);
            padding: 36px;
            width: 100%;
            max-width: 520px;
        }

        /* Drop zone */
        .drop-zone {
            border: 2px dashed var(--border);
            border-radius: 12px;
            padding: 36px 20px;
            text-align: center;
            cursor: pointer;
            transition: border-color 0.2s, background 0.2s;
            position: relative;
            margin-bottom: 20px;
        }
        .drop-zone:hover,
        .drop-zone.drag-over { border-color: var(--blue); background: #eff6ff; }
        .drop-zone.has-file   { border-color: var(--green); background: #f0fdf4; }
        .drop-zone input[type="file"] {
            position: absolute; inset: 0; opacity: 0; cursor: pointer;
        }
        .drop-icon {
            font-size: 2.8rem;
            margin-bottom: 10px;
            line-height: 1;
        }
        .drop-label { font-weight: 600; font-size: 1rem; }
        .drop-hint  { color: var(--muted); font-size: 0.82rem; margin-top: 4px; }
        .file-name  { color: var(--green); font-weight: 600; margin-top: 10px; font-size: 0.9rem; }

        /* Preview */
        #previewWrap {
            display: none;
            margin-bottom: 20px;
            border-radius: 10px;
            overflow: hidden;
            border: 1px solid var(--border);
        }
        #previewWrap.show { display: block; }
        #preview {
            width: 100%;
            max-height: 280px;
            object-fit: contain;
            display: block;
            background: #f1f5f9;
        }

        /* Submit button */
        .btn {
            display: block;
            width: 100%;
            padding: 14px;
            border: none;
            border-radius: 10px;
            background: var(--blue);
            color: #fff;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: background 0.2s, transform 0.1s;
        }
        .btn:hover:not(:disabled) { background: var(--blue-dark); }
        .btn:active:not(:disabled) { transform: scale(0.98); }
        .btn:disabled { background: var(--muted); cursor: not-allowed; }

        /* Loading overlay */
        .loading {
            display: none;
            flex-direction: column;
            align-items: center;
            margin-top: 24px;
        }
        .loading.show { display: flex; }
        .spinner {
            width: 44px; height: 44px;
            border: 4px solid var(--border);
            border-top-color: var(--blue);
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
            margin-bottom: 12px;
        }
        @keyframes spin { to { transform: rotate(360deg); } }
        .loading p { color: var(--muted); font-size: 0.9rem; }

        /* Error */
        .error-box {
            margin-top: 16px;
            background: #fef2f2;
            border: 1px solid #fecaca;
            color: #dc2626;
            border-radius: 8px;
            padding: 12px 16px;
            font-size: 0.9rem;
        }
    </style>
</head>
<body>
    <header>
        <h1>AI Slope Detector</h1>
        <p>Upload a street-view photo to measure slope and check AODA compliance</p>
    </header>

    <div class="card">
        <form id="uploadForm" action="/analyze" method="post" enctype="multipart/form-data">
            <div class="drop-zone" id="dropZone">
                <input type="file" id="fileInput" name="image" accept="image/jpeg,image/png" />
                <div class="drop-icon">🖼️</div>
                <p class="drop-label">Drag &amp; drop an image here</p>
                <p class="drop-hint">or click to browse &nbsp;·&nbsp; JPG / PNG &nbsp;·&nbsp; max 16 MB</p>
                <p class="file-name" id="fileName"></p>
            </div>

            <div id="previewWrap">
                <img id="preview" alt="Preview" />
            </div>

            <button class="btn" type="submit" id="submitBtn" disabled>Analyze Slope</button>
        </form>

        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p>Running pipeline… this may take 30–60 seconds</p>
        </div>

        {% if error %}
        <div class="error-box">{{ error }}</div>
        {% endif %}
    </div>

    <script>
        const fileInput   = document.getElementById('fileInput');
        const dropZone    = document.getElementById('dropZone');
        const fileNameEl  = document.getElementById('fileName');
        const previewWrap = document.getElementById('previewWrap');
        const preview     = document.getElementById('preview');
        const submitBtn   = document.getElementById('submitBtn');
        const form        = document.getElementById('uploadForm');
        const loading     = document.getElementById('loading');

        function handleFile(file) {
            if (!file) return;
            fileNameEl.textContent = file.name;
            dropZone.classList.add('has-file');
            submitBtn.disabled = false;

            const reader = new FileReader();
            reader.onload = e => {
                preview.src = e.target.result;
                previewWrap.classList.add('show');
            };
            reader.readAsDataURL(file);
        }

        fileInput.addEventListener('change', () => handleFile(fileInput.files[0]));

        dropZone.addEventListener('dragover', e => {
            e.preventDefault();
            dropZone.classList.add('drag-over');
        });
        dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
        dropZone.addEventListener('drop', e => {
            e.preventDefault();
            dropZone.classList.remove('drag-over');
            const dt = e.dataTransfer;
            if (dt.files.length) {
                // Push file into input so form submission works
                const transfer = new DataTransfer();
                transfer.items.add(dt.files[0]);
                fileInput.files = transfer.files;
                handleFile(dt.files[0]);
            }
        });

        form.addEventListener('submit', () => {
            submitBtn.disabled = true;
            loading.classList.add('show');
        });
    </script>
</body>
</html>
"""

RESULT_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Slope Detection Results</title>
    <style>
        :root {
            --bg: #f8fafc;
            --card: #ffffff;
            --border: #e2e8f0;
            --text: #1e293b;
            --muted: #94a3b8;
            --blue: #3b82f6;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: var(--bg);
            color: var(--text);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 40px 24px;
        }
        header {
            text-align: center;
            margin-bottom: 28px;
        }
        header h1 { font-size: 1.8rem; font-weight: 700; }
        .card {
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 16px;
            box-shadow: 0 4px 24px rgba(0,0,0,0.06);
            padding: 36px;
            width: 100%;
            max-width: 580px;
        }

        /* AODA status badge */
        .status-block {
            text-align: center;
            margin-bottom: 28px;
        }
        .status-label {
            font-size: 0.8rem;
            font-weight: 600;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: var(--muted);
            margin-bottom: 8px;
        }
        .status-badge {
            display: inline-block;
            padding: 10px 32px;
            border-radius: 999px;
            font-weight: 700;
            font-size: 1.15rem;
        }
        .status-PASS        { background: #dcfce7; color: #166534; }
        .status-CONDITIONAL { background: #fef9c3; color: #854d0e; }
        .status-FAIL        { background: #fee2e2; color: #991b1b; }

        /* Gauge */
        .gauge-wrap {
            margin: 0 auto 28px;
            width: 200px;
            text-align: center;
        }
        .gauge-wrap svg { overflow: visible; }
        .gauge-track { fill: none; stroke: #e2e8f0; stroke-width: 14; }
        .gauge-arc   { fill: none; stroke-width: 14; stroke-linecap: round;
                       transition: stroke-dasharray 1s ease; }
        .gauge-text  { font-size: 1.3rem; font-weight: 700; fill: var(--text); }
        .gauge-sub   { font-size: 0.65rem; fill: var(--muted); }

        /* Metrics table */
        .metrics { width: 100%; border-collapse: collapse; margin-bottom: 24px; }
        .metrics tr:not(:last-child) td { border-bottom: 1px solid var(--border); }
        .metrics td { padding: 13px 4px; }
        .metrics td:first-child { color: var(--muted); font-size: 0.9rem; }
        .metrics td:last-child  { font-weight: 600; text-align: right; }

        /* Warnings */
        .warnings {
            background: #fffbeb;
            border-left: 4px solid #f59e0b;
            border-radius: 6px;
            padding: 12px 16px;
            margin-bottom: 24px;
            font-size: 0.88rem;
            line-height: 1.6;
        }
        .warnings strong { display: block; margin-bottom: 4px; color: #92400e; }

        /* Image filename chip */
        .filename-chip {
            display: inline-block;
            background: #f1f5f9;
            border-radius: 6px;
            padding: 2px 8px;
            font-size: 0.82rem;
            color: var(--muted);
            font-family: monospace;
        }

        /* Back button */
        .btn-back {
            display: block;
            text-align: center;
            background: var(--blue);
            color: #fff;
            text-decoration: none;
            padding: 14px;
            border-radius: 10px;
            font-size: 1rem;
            font-weight: 600;
            transition: background 0.2s;
        }
        .btn-back:hover { background: #2563eb; }
    </style>
</head>
<body>
    <header><h1>Slope Detection Results</h1></header>

    <div class="card">

        <!-- AODA badge -->
        <div class="status-block">
            <p class="status-label">AODA Compliance</p>
            <span class="status-badge status-{{ results.aoda_status }}">
                {{ results.aoda_status }}
            </span>
        </div>

        <!-- Gauge -->
        <div class="gauge-wrap">
            <svg viewBox="0 0 200 120" width="200" height="120">
                <!-- 180° arc from left to right, centre (100,100) r=80 -->
                <!-- Track -->
                <path class="gauge-track"
                      d="M 20,100 A 80,80 0 0,1 180,100" />
                <!-- Coloured arc -->
                <path class="gauge-arc" id="gaugeArc"
                      d="M 20,100 A 80,80 0 0,1 180,100"
                      stroke="#22c55e"
                      stroke-dasharray="0 251.3" />
                <!-- Labels -->
                <text class="gauge-text" x="100" y="95" text-anchor="middle" id="gaugeVal">
                    {{ "%.1f"|format(results.slope_degrees) }}°
                </text>
                <text class="gauge-sub" x="100" y="112" text-anchor="middle">
                    slope angle
                </text>
            </svg>
        </div>

        <!-- Metrics -->
        <table class="metrics">
            <tr>
                <td>Slope angle</td>
                <td>{{ "%.2f"|format(results.slope_degrees) }}°</td>
            </tr>
            <tr>
                <td>Slope grade</td>
                <td>{{ "%.2f"|format(results.slope_percent_grade) }}%</td>
            </tr>
            <tr>
                <td>Plane fit inlier ratio</td>
                <td>{{ "%.3f"|format(results.inlier_ratio) }}</td>
            </tr>
            <tr>
                <td>Scale factor</td>
                <td>{{ "%.6f"|format(results.scale_factor) }}</td>
            </tr>
            <tr>
                <td>Scale method</td>
                <td>{{ results.scale_method }}</td>
            </tr>
            <tr>
                <td>Image</td>
                <td><span class="filename-chip">{{ filename }}</span></td>
            </tr>
        </table>

        {% if results.warnings %}
        <div class="warnings">
            <strong>Warnings</strong>
            {% for w in results.warnings %}
            • {{ w }}<br>
            {% endfor %}
        </div>
        {% endif %}

        <a href="/" class="btn-back">← Analyze Another Image</a>
    </div>

    <script>
        // Animate the gauge arc
        const degrees    = {{ results.slope_degrees }};
        const maxDegrees = 15;   // treat 15° as the 100% end of gauge
        const ARC_LEN    = 251.3; // half circumference of r=80 circle

        const pct  = Math.min(degrees / maxDegrees, 1);
        const fill = pct * ARC_LEN;
        const arc  = document.getElementById('gaugeArc');

        // colour based on AODA status
        const status = "{{ results.aoda_status }}";
        const colour = status === 'PASS' ? '#22c55e'
                     : status === 'FAIL' ? '#ef4444'
                     : '#f59e0b';
        arc.setAttribute('stroke', colour);

        // Trigger animation after paint
        requestAnimationFrame(() => {
            arc.style.transition = 'stroke-dasharray 1.2s cubic-bezier(.4,0,.2,1)';
            arc.setAttribute('stroke-dasharray', fill + ' ' + (ARC_LEN - fill));
        });
    </script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/", methods=["GET"])
def index():
    return render_template_string(INDEX_HTML, error=None)


@app.route("/analyze", methods=["POST"])
def analyze():
    file = request.files.get("image")
    if file is None or file.filename == "":
        return render_template_string(INDEX_HTML, error="Please select an image file.")

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in (".jpg", ".jpeg", ".png"):
        return render_template_string(INDEX_HTML, error="Only JPG and PNG files are supported.")

    safe_name = f"{uuid.uuid4().hex}{ext}"
    filepath = os.path.join(UPLOAD_FOLDER, safe_name)
    file.save(filepath)

    try:
        results = run_pipeline(filepath, output_json=None)
        return render_template_string(
            RESULT_HTML,
            results=results,
            filename=file.filename,
        )
    except Exception as e:
        return render_template_string(INDEX_HTML, error=f"Analysis failed: {e}")
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


if __name__ == "__main__":
    print("Starting AI Slope Detector web app...")
    print("Open http://127.0.0.1:5000 in your browser")
    app.run(debug=False, host="127.0.0.1", port=5000)
