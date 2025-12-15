# src/neural_network/plot_loss_curve.py
#
# Citește results/training_history.csv și salvează docs/loss_curve.png
# (cerință Nivel 2 în README)

import os
import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = "results/training_history.csv"
OUT_PNG = "docs/loss_curve.png"

def main():
    os.makedirs("docs", exist_ok=True)

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Nu există: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)

    # În results.csv Ultralytics, coloanele uzuale sunt: train/box_loss, val/box_loss etc.
    # Noi plotăm ce găsim relevant.
    train_cols = [c for c in df.columns if c.startswith("train/")]
    val_cols = [c for c in df.columns if c.startswith("val/")]

    # Preferăm box_loss ca principal pentru detecție:
    train_key = "train/box_loss" if "train/box_loss" in df.columns else (train_cols[0] if train_cols else None)
    val_key = "val/box_loss" if "val/box_loss" in df.columns else (val_cols[0] if val_cols else None)

    if train_key is None or val_key is None:
        raise RuntimeError(f"Nu găsesc coloane train/ și val/ în {CSV_PATH}. Coloane: {list(df.columns)}")

    plt.figure()
    plt.plot(df[train_key], label=train_key)
    plt.plot(df[val_key], label=val_key)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss vs Val Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=200)
    plt.close()

    print("[INFO] Grafic salvat în:", OUT_PNG)

if __name__ == "__main__":
    main()
