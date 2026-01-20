import json
from pathlib import Path
from ultralytics import YOLO
import numpy as np

model_path = "models/optimized_model.pt"
data_yaml = "config/car_damage.yaml"

model = YOLO(model_path)
r = model.val(data=data_yaml, imgsz=640, plots=False)

# r.box.p si r.box.r pot fi vectori (pe clase). Le convertim in scalari.
def to_float(x):
    if x is None:
        return 0.0
    if isinstance(x, (float, int)):
        return float(x)
    try:
        arr = np.array(x, dtype=float)
        if arr.ndim == 0:
            return float(arr)
        # macro average (medie peste clase)
        return float(np.nanmean(arr))
    except Exception:
        return 0.0

metrics = {
    "model": model_path,
    "imgsz": 640,
    # macro-averaged P/R (aproape de 'all' din tabel, suficient pentru raport)
    "precision_macro": to_float(getattr(r.box, "p", None)),
    "recall_macro": to_float(getattr(r.box, "r", None)),
    # mAP-urile sunt scalare in Ultralytics
    "mAP50": float(getattr(r.box, "map50", 0.0)),
    "mAP50_95": float(getattr(r.box, "map", 0.0)),
}

Path("results").mkdir(exist_ok=True)
out = Path("results/final_metrics.json")
out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

print("✔ Wrote:", out)
print(json.dumps(metrics, indent=2))