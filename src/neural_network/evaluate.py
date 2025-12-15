# src/neural_network/evaluate.py
#
# Scop:
#  - Evaluează modelul antrenat pe TEST set (din config/car_damage.yaml)
#  - Salvează metricile în results/test_metrics.json (cerință README)
#
# Notă:
#  - Pentru detecție YOLO, raportăm P/R/mAP și calculăm și F1 (macro aproximat pe baza P/R globale).

import os
import json
from datetime import datetime

from ultralytics import YOLO

MODEL_PATH = "models/trained_model.pt"
DATA_CONFIG = "config/car_damage.yaml"
OUT_JSON = "results/test_metrics.json"

def safe_div(a, b):
    return a / b if b != 0 else 0.0

def main():
    os.makedirs("results", exist_ok=True)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Nu există modelul: {MODEL_PATH}")

    if not os.path.exists(DATA_CONFIG):
        raise FileNotFoundError(f"Nu există config-ul dataset: {DATA_CONFIG}")

    print("[INFO] Încarc modelul:", MODEL_PATH)
    model = YOLO(MODEL_PATH)

    print("[INFO] Rulez evaluarea pe TEST set conform YAML:", DATA_CONFIG)
    # split="test" forțează evaluarea pe test set
    r = model.val(data=DATA_CONFIG, split="test")

    # Ultralytics returnează un obiect Results/metrics. Extragem ce e stabil:
    # box precision/recall/mAP
    # Denumirile pot diferi ușor între versiuni, deci încercăm robust.
    metrics = getattr(r, "box", None)

    if metrics is None:
        raise RuntimeError("Nu am găsit r.box metrics. Verifică versiunea Ultralytics/YOLO.")

    P = float(getattr(metrics, "mp", 0.0))      # mean precision
    R = float(getattr(metrics, "mr", 0.0))      # mean recall
    mAP50 = float(getattr(metrics, "map50", 0.0))
    mAP5095 = float(getattr(metrics, "map", 0.0))  # mAP50-95

    F1 = safe_div(2 * P * R, (P + R))

    payload = {
        "evaluated_model": MODEL_PATH,
        "data_config": DATA_CONFIG,
        "split": "test",
        "box_precision_mean": P,
        "box_recall_mean": R,
        "box_f1_mean": F1,
        "map50": mAP50,
        "map50_95": mAP5095,
        "created_at": datetime.now().isoformat(timespec="seconds")
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("[INFO] Metrici salvate în:", OUT_JSON)
    print("[INFO] P=", P, "R=", R, "F1=", F1, "mAP50=", mAP50, "mAP50-95=", mAP5095)

if __name__ == "__main__":
    main()
