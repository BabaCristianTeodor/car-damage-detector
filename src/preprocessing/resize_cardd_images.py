# src/preprocessing/resize_cardd_images.py
#
# scop: redimensionăm imaginile din data/raw/cardd (train/val/test)
#       astfel încât latura cea mai mare să fie 1280 px.
#       imaginile procesate le salvăm în data/processed/cardd_resized,
#       păstrând structura pe split-uri.

from pathlib import Path
from PIL import Image

# rădăcina repo-ului
BASE_DIR = Path(__file__).resolve().parents[2]

# imagini originale
RAW_ROOT = BASE_DIR / "data" / "raw" / "cardd"

# imagini redimensionate
OUT_ROOT = BASE_DIR / "data" / "processed" / "cardd_resized"

# latura maximă dorită
MAX_SIZE = 1280


def resize_image(src_path: Path, dst_path: Path):
    """deschide o imagine, o rescalează păstrând proporțiile, apoi o salvează."""
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(src_path) as img:
        w, h = img.size

        # dacă deja imaginea e „mică”, o lăsăm așa
        if max(w, h) <= MAX_SIZE:
            img.save(dst_path)
            return

        # factor de scalare astfel încât latura mare să devină MAX_SIZE
        scale = MAX_SIZE / max(w, h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        img_resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        img_resized.save(dst_path)


def process_split(split: str):
    """
    procesează un singur split (train / val / test).
    citește din data/raw/cardd/<split>
    scrie în data/processed/cardd_resized/<split>/images
    """
    src_dir = RAW_ROOT / split
    dst_dir = OUT_ROOT / split / "images"

    if not src_dir.exists():
        print(f"[WARN] Folderul sursă nu există: {src_dir}")
        return

    print(f"\n=== REDIMENSIONARE imagini pentru split-ul: {split} ===")
    count = 0
    # dacă vrei și .png/.jpeg, poți extinde lista de extensii
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        for img_path in src_dir.glob(ext):
            dst_path = dst_dir / img_path.name
            resize_image(img_path, dst_path)
            count += 1

    print(f"[OK] Am procesat {count} imagini pentru {split}. Rezultatul este în {dst_dir}")


def main():
    for split in ["train", "val", "test"]:
        process_split(split)


if __name__ == "__main__":
    main()
