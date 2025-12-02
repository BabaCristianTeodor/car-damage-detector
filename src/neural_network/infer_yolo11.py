from ultralytics import YOLO
from pathlib import Path

def main():
    repo_root = Path(__file__).resolve().parents[2]
    model_path = repo_root / "models" / "yolo11-cardd.pt"

    model = YOLO(str(model_path))

    # pune aici o poză de test, de ex. din data/test/images
    img_path = repo_root / "data" / "test" / "images" / "003995.png"

    results = model.predict(source=str(img_path), save=True, project=str(repo_root / "runs"), name="manual_test", exist_ok=True)
    print("Rezultatul salvat în:", results[0].save_dir)

if __name__ == "__main__":
    main()
