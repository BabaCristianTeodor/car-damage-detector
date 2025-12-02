# src/preprocessing/enhance_cardd_images.py
#
# scop: ia imaginile redimensionate din data/processed/cardd_resized
#       și le face puțin mai „clare” pentru damage:
#       - crește contrastul cu ~15%
#       - crește luminozitatea cu ~5%
#       salvează rezultatul în data/processed/cardd_enhanced.

from pathlib import Path
from PIL import Image, ImageEnhance

BASE_DIR = Path(__file__).resolve().parents[2]

# imagini redimensionate
RESIZED_ROOT = BASE_DIR / "data" / "processed" / "cardd_resized"

# imagini redimensionate + enhance
OUT_ROOT = BASE_DIR / "data" / "processed" / "cardd_enhanced"

# factori de ajustare (nu exagerăm)
CONTRAST_FACTOR = 1.15   # +15% contrast
BRIGHTNESS_FACTOR = 1.05 # +5% luminozitate


def enhance_image(src_path: Path, dst_path: Path):
    """aplică un tweak mic de contrast + luminozitate unei imagini."""
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(src_path) as img:
        # contrast
        enhancer_contrast = ImageEnhance.Contrast(img)
        img_c = enhancer_contrast.enhance(CONTRAST_FACTOR)

        # luminozitate
        enhancer_brightness = ImageEnhance.Brightness(img_c)
        img_final = enhancer_brightness.enhance(BRIGHTNESS_FACTOR)

        img_final.save(dst_path)


def process_split(split: str):
    """
    procesează un split (train / val / test).
    citește din data/processed/cardd_resized/<split>/images
    scrie în data/processed/cardd_enhanced/<split>/images
    """
    src_dir = RESIZED_ROOT / split / "images"
    dst_dir = OUT_ROOT / split / "images"

    if not src_dir.exists():
        print(f"[WARN] Folderul sursă nu există: {src_dir}")
        return

    print(f"\n=== ENHANCE imagini pentru split-ul: {split} ===")
    count = 0
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        for img_path in src_dir.glob(ext):
            dst_path = dst_dir / img_path.name
            enhance_image(img_path, dst_path)
            count += 1

    print(f"[OK] Am procesat {count} imagini pentru {split}. Rezultatul este în {dst_dir}")


def main():
    for split in ["train", "val", "test"]:
        process_split(split)


if __name__ == "__main__":
    main()
