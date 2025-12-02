# src/web/app.py
import uuid
from pathlib import Path

import cv2
from flask import Flask, render_template, request, redirect, url_for
from ultralytics import YOLO
from werkzeug.utils import secure_filename

# Rădăcina repo-ului (car-damage-detector/)
BASE_DIR = Path(__file__).resolve().parents[2]

# Directorul web (src/web/)
WEB_DIR = Path(__file__).resolve().parent

# Foldere pentru upload + rezultate în static/
UPLOAD_DIR = WEB_DIR / "static" / "uploads"
RESULT_DIR = WEB_DIR / "static" / "results"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# Modelul tău antrenat
MODEL_PATH = BASE_DIR / "models" / "yolo11-cardd.pt"

app = Flask(
    __name__,
    static_folder="static",
    template_folder="templates",
)

# Încărcăm modelul o singură dată
model = YOLO(str(MODEL_PATH))


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "image" not in request.files:
            return redirect(request.url)

        file = request.files["image"]
        if not file or file.filename == "":
            return redirect(request.url)

        # ===== 1. Salvăm imaginea uploadată în uploads/ cu un nume unic =====
        orig_ext = Path(file.filename).suffix.lower()
        safe_name = secure_filename(Path(file.filename).stem)
        unique_id = uuid.uuid4().hex[:8]
        input_filename = f"{safe_name}_{unique_id}{orig_ext}"
        upload_path = UPLOAD_DIR / input_filename
        file.save(upload_path)

        # ===== 2. Rulăm YOLO pe imagine (fără save=True) =====
        results = model.predict(
            source=str(upload_path),
            imgsz=768,
            conf=0.25,
            iou=0.5,
            save=False,      # NU mai lăsăm YOLO să salveze în runs/
            verbose=False,
        )

        # ===== 3. Salvăm noi imaginea adnotată în results/ =====
        output_filename = f"{safe_name}_{unique_id}_pred{orig_ext}"
        result_path = RESULT_DIR / output_filename

        # results[0].plot() -> imagine cu bounding-box-uri (numpy array BGR)
        annotated = results[0].plot()
        cv2.imwrite(str(result_path), annotated)

        # ===== 4. Trimitem căile către template =====
        return render_template(
            "index.html",
            original_image=url_for("static", filename=f"uploads/{input_filename}"),
            result_image=url_for("static", filename=f"results/{output_filename}"),
        )

    # GET – pagina fără imagini
    return render_template("index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
