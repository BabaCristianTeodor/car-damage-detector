# src/neural_network/train_yolo11_seg.py
from ultralytics import YOLO
from pathlib import Path
from shutil import copy2


def main():
    # repo_root = car-damage-detector/
    repo_root = Path(__file__).resolve().parents[2]
    data_yaml = repo_root / "config" / "cardd_seg_yolo.yaml"

    models_dir = repo_root / "models"
    models_dir.mkdir(exist_ok=True)

    # aici folosim modelul de baza pentru segmentare
    model = YOLO("yolo11m-seg.pt")

    results = model.train(
        data=str(data_yaml),
        epochs=200,          # mai mult timp de învățare, cum ai zis
        imgsz=768,          # puțin mai mare, să prindă zgârieturi subțiri
        batch=8,
        project=str(repo_root / "runs"),
        name="yolo11-cardd-seg-200ep",
        # augmentare un pic mai agresivă, tocmai pentru cazurile tale dificile
        hsv_h=0.1,
        hsv_s=0.8,
        hsv_v=0.7,
        scale=0.5,
        mixup=0.2,
        erasing=0.4,
    )

    run_dir = Path(results.save_dir)
    best_pt = run_dir / "weights" / "best.pt"
    if best_pt.exists():
        dst = models_dir / "yolo11-cardd-seg.pt"
        copy2(best_pt, dst)
        print(f"[OK] Modelul de segmentare cel mai bun a fost copiat în {dst}")
    else:
        print("[WARN] Nu am găsit best.pt pentru segmentare, verifică runs/ manual.")

    print("\n[DONE] Antrenare YOLO11 SEG terminată.")


if __name__ == "__main__":
    main()
