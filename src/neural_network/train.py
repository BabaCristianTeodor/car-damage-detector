# src/neural_network/train.py
#
# TRAIN YOLO (versiune simplă, fără argumente)
# - Citește automat dataset-ul din config/car_damage.yaml
# - Antrenează YOLO11m cu setări mai bune pentru clase subtile (crack/scratch)
# - Salvează exact fișierele cerute:
#     models/trained_model.pt
#     results/training_history.csv
#     results/hyperparameters.yaml

import os
import shutil
from datetime import datetime

import yaml
from ultralytics import YOLO


# =========================
# CONFIG FIX (NU MAI DAI ARGUMENTE)
# =========================

DATA_CONFIG = "config/car_damage.yaml"
BASE_MODEL = "yolo11m.pt"

# Setări recomandate pentru RTX 4060 8GB:
EPOCHS = 100
BATCH_SIZE = 8
IMG_SIZE = 960

# Early stopping: oprește dacă nu se îmbunătățește validarea timp de N epoci
EARLY_STOPPING_PATIENCE = 25

# Unde ține Ultralytics log-urile / run-urile
PROJECT_NAME = "runs/rn_train"
RUN_NAME = "car_damage_yolo_v2"

# Augmentări (safe, utile fără poze noi)
AUGMENT = {
    "fliplr": 0.5,       # flip orizontal cu probabilitate 50%
    "hsv_h": 0.015,      # variație mică de hue
    "hsv_s": 0.7,        # variație saturare
    "hsv_v": 0.4,        # variație luminozitate
    "translate": 0.1,    # mică translație
    "scale": 0.5,        # scalare (zoom in/out)
    "mosaic": 1.0,       # mosaic activ (ajută mult la generalizare)
}


# =========================
# UTILS
# =========================

def ensure_directories():
    """Creăm folderele standard cerute de proiect."""
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)


def copy_best_model(run_dir: str):
    """
    Copiem best.pt în models/trained_model.pt
    (modelul final pe care îl folosești la evaluare/demonstrație).
    """
    src_best = os.path.join(run_dir, "weights", "best.pt")
    dst_best = os.path.join("models", "trained_model.pt")

    if not os.path.exists(src_best):
        print(f"[AVERTISMENT] Nu am găsit best.pt la: {src_best}")
        return

    shutil.copy2(src_best, dst_best)
    print(f"[INFO] trained_model.pt salvat în: {dst_best}")


def copy_training_history(run_dir: str):
    """
    Ultralytics YOLO salvează automat results.csv în run_dir.
    Îl copiem în results/training_history.csv ca să ai fix fișierul cerut.
    """
    src_csv = os.path.join(run_dir, "results.csv")
    dst_csv = os.path.join("results", "training_history.csv")

    if not os.path.exists(src_csv):
        print(f"[AVERTISMENT] Nu am găsit results.csv la: {src_csv}")
        return

    shutil.copy2(src_csv, dst_csv)
    print(f"[INFO] training_history.csv salvat în: {dst_csv}")


def save_hyperparameters(run_dir: str):
    """
    Salvăm hiperparametrii folosiți la train într-un YAML ușor de prezentat.
    """
    payload = {
        "data_config": DATA_CONFIG,
        "base_model": BASE_MODEL,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "img_size": IMG_SIZE,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "augmentations": AUGMENT,
        "yolo_run_dir": run_dir,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }

    out_path = os.path.join("results", "hyperparameters.yaml")
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False)

    print(f"[INFO] Hiperparametri salvați în: {out_path}")


def print_config():
    print("=== TRAIN YOLO – CONFIG (AUTO) ===")
    print(f"[INFO] DATA_CONFIG  = {DATA_CONFIG}")
    print(f"[INFO] BASE_MODEL   = {BASE_MODEL}")
    print(f"[INFO] EPOCHS       = {EPOCHS}")
    print(f"[INFO] BATCH_SIZE   = {BATCH_SIZE}")
    print(f"[INFO] IMG_SIZE     = {IMG_SIZE}")
    print(f"[INFO] PATIENCE     = {EARLY_STOPPING_PATIENCE}")
    print(f"[INFO] PROJECT_NAME = {PROJECT_NAME}")
    print(f"[INFO] RUN_NAME     = {RUN_NAME}")
    print(f"[INFO] AUGMENT      = {AUGMENT}")
    print("=================================\n")


# =========================
# MAIN
# =========================

def main():
    ensure_directories()
    print_config()

    # 1) Încărcăm modelul YOLO de bază
    model = YOLO(BASE_MODEL)

    # 2) Pornim train-ul
    # NOTE: augmentările sunt transmise ca parametri direcți
    train_kwargs = dict(
        data=DATA_CONFIG,
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMG_SIZE,
        patience=EARLY_STOPPING_PATIENCE,
        project=PROJECT_NAME,
        name=RUN_NAME,
        exist_ok=True,
    )
    train_kwargs.update(AUGMENT)

    _ = model.train(**train_kwargs)

    # 3) Directorul în care YOLO a salvat acest run
    run_dir = str(model.trainer.save_dir)
    print(f"[INFO] YOLO a salvat run-ul în: {run_dir}")

    # 4) Salvăm ce cere proiectul (models/ + results/)
    copy_best_model(run_dir)
    copy_training_history(run_dir)
    save_hyperparameters(run_dir)

    print("\n[INFO] Antrenarea s-a terminat.")
    print("[INFO] Output proiect:")
    print("       - models/trained_model.pt")
    print("       - results/training_history.csv")
    print("       - results/hyperparameters.yaml")


if __name__ == "__main__":
    main()
