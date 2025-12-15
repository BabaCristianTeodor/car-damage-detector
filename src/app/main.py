# src/app/main.py
import io
import os
import time
import base64
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
from PIL import Image
import cv2

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from ultralytics import YOLO

APP_TITLE = "Car Damage Detector — Academic Demo"
MODEL_PATH = "models/trained_model.pt"

app = FastAPI(title=APP_TITLE)
app.mount("/static", StaticFiles(directory="src/app/static"), name="static")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Nu găsesc modelul la {MODEL_PATH}. Asigură-te că ai models/trained_model.pt")

model = YOLO(MODEL_PATH)


def read_image_to_numpy_rgb(upload: UploadFile) -> np.ndarray:
    content = upload.file.read()
    img = Image.open(io.BytesIO(content)).convert("RGB")
    return np.array(img)

def enhance_image_bgr(img_bgr: np.ndarray) -> np.ndarray:
    """
    Enhance SOFT (recomandat pentru inferență):
      - CLAHE blând pe canalul L (LAB) -> evidențiază ușor detaliile
      - gamma discret -> ridică ușor umbrele fără să ardă highlights
      - sharpen foarte mic (unsharp weak) -> doar pentru micro-contrast
    Scop: să nu "deformeze" distribuția imaginii, ca să nu scadă detecția.
    """
    # 1) CLAHE blând (mai puțin agresiv decât înainte)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8, 8))
    l2 = clahe.apply(l)

    lab2 = cv2.merge((l2, a, b))
    out = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

    # 2) Gamma corection discret (1.0 = no change)
    gamma = 0.95  # <1.0: luminează ușor; >1.0: întunecă
    inv = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv) * 255 for i in range(256)]).astype("uint8")
    out = cv2.LUT(out, table)

    # 3) Sharpen FOARTE mic (ca să nu introducă artefacte)
    blurred = cv2.GaussianBlur(out, (0, 0), sigmaX=0.8, sigmaY=0.8)
    out = cv2.addWeighted(out, 1.08, blurred, -0.08, 0)

    return out

def png_bytes_from_rgb(rgb: np.ndarray) -> bytes:
    img = Image.fromarray(rgb)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def to_data_url(png_bytes: bytes) -> str:
    b64 = base64.b64encode(png_bytes).decode("ascii")
    return f"data:image/png;base64,{b64}"


@app.get("/", response_class=HTMLResponse)
def index():
    return """
<!doctype html>
<html lang="ro">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Car Damage Detector — Academic Demo</title>
  <link rel="stylesheet" href="/static/app.css" />
</head>
<body>
  <div class="bg"></div>

  <header class="topbar">
    <div class="brand">
      <div class="mark"></div>
      <div class="brand-text">
        <div class="title">Car Damage Detector</div>
        <div class="subtitle">Academic Demo • YOLO11m • Inference local</div>
      </div>
    </div>
    <div class="chip">Model: <b>trained_model.pt</b></div>
  </header>

    <main class="shell">
    <section class="panel glass">
      <div class="panel-top">
        <div class="header-row">
          <div class="copy">
            <h1>Detecție daune auto, cu inferență reală.</h1>
            <p>Încarcă o imagine. Sistemul generează o variantă Enhanced și rulează YOLO, întorcând imaginea Annotated.</p>
          </div>

          <div class="controls">
            <label class="btn primary">
              <input id="file" type="file" accept="image/*" hidden />
              Upload image
            </label>
            <button id="run" class="btn" disabled>Run inference</button>
            <button id="clear" class="btn ghost" disabled>Clear</button>
          </div>
        </div>

        <div class="stats-grid">
          <div class="stat glass-soft">
            <div class="left">
              <div class="k">Status</div>
              <div class="v" id="statusText">Idle</div>
            </div>
            <div class="pill idle" id="statusPill">Idle</div>
          </div>

          <div class="stat glass-soft">
            <div class="left">
              <div class="k">Detections</div>
              <div class="v" id="detCount">—</div>
            </div>
            <div class="pill" id="detPill">Objects</div>
          </div>

          <div class="stat glass-soft">
            <div class="left">
              <div class="k">Latency</div>
              <div class="v" id="latency">—</div>
            </div>
            <div class="pill" id="latPill">ms</div>
          </div>
        </div>

        <div class="classes glass-soft">
          <div class="classes-title">Detected classes</div>
          <div id="classes" class="tags"></div>
        </div>
      </div>

      <div class="panel-bottom">
        <div class="img-grid">
          <div class="img-card glass-soft">
            <div class="img-head">
              <div>
                <div class="img-title">Enhanced</div>
                <div class="img-sub">Pre-processing pentru evidențierea defectelor</div>
              </div>
              <div class="pill">Click to zoom</div>
            </div>

            <div class="viewer" data-view="enhanced">
              <img id="enhanced" alt="Enhanced output" />
              <div class="empty" id="enhEmpty">Încarcă o imagine și apasă Run.</div>
            </div>
          </div>

          <div class="img-card glass-soft">
            <div class="img-head">
              <div>
                <div class="img-title">Annotated</div>
                <div class="img-sub">Predicții YOLO (bounding boxes)</div>
              </div>
              <div class="pill">Click to zoom</div>
            </div>

            <div class="viewer" data-view="annotated">
              <img id="annotated" alt="Annotated output" />
              <div class="empty" id="annEmpty">Încarcă o imagine și apasă Run.</div>
            </div>
          </div>
        </div>
      </div>
    </section>
  </main>

  <div class="overlay" id="overlay">
    <div class="overlay-card">
      <div class="overlay-head">
        <div class="overlay-title" id="overlayTitle">Preview</div>
        <button class="overlay-close" id="overlayClose">Close</button>
      </div>
      <div class="overlay-body">
        <img id="overlayImg" alt="Zoomed preview" />
      </div>
    </div>
  </div>

  <script src="/static/app.js"></script>
</body>
</html>
"""


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)) -> JSONResponse:
    """
    Returnează:
      - enhanced (PNG data-url)
      - annotated (PNG data-url)
      - meta (detections, latency, items)
    """
    t0 = time.time()

    rgb = read_image_to_numpy_rgb(file)
    bgr = rgb[..., ::-1].copy()

    # 1) Enhanced
    enhanced_bgr = enhance_image_bgr(bgr)
    enhanced_rgb = enhanced_bgr[..., ::-1]
    enhanced_png = png_bytes_from_rgb(enhanced_rgb)

    # 2) Predict pe imaginea enhanced (recomandat, consistent cu ce vrei tu)
    results = model.predict(enhanced_rgb, verbose=False)
    r0 = results[0]

    annotated_bgr = r0.plot()
    annotated_rgb = annotated_bgr[..., ::-1]
    annotated_png = png_bytes_from_rgb(annotated_rgb)

    # 3) Meta detecții
    names = r0.names
    boxes = getattr(r0, "boxes", None)

    dets: List[Dict[str, Any]] = []
    if boxes is not None and len(boxes) > 0:
        cls = boxes.cls.cpu().numpy().astype(int)
        conf = boxes.conf.cpu().numpy()
        xyxy = boxes.xyxy.cpu().numpy()
        for i in range(len(cls)):
            dets.append({
                "class_id": int(cls[i]),
                "class_name": str(names[int(cls[i])]),
                "confidence": float(conf[i]),
                "xyxy": [float(x) for x in xyxy[i].tolist()]
            })

    latency_ms = round((time.time() - t0) * 1000.0, 2)

    payload = {
        "filename": file.filename,
        "detections": len(dets),
        "latency_ms": latency_ms,
        "items": dets,
        "enhanced_png": to_data_url(enhanced_png),
        "annotated_png": to_data_url(annotated_png),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }

    return JSONResponse(payload)
