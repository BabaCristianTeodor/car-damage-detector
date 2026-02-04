# src/preprocessing/enhance_from_data_images.py
#
# Structură de INPUT (ce ai TU acum):
#
#   data/
#     images/
#       train/
#         images/   <- JPG/PNG
#         labels/   <- TXT (YOLO)
#       val/
#         images/
#         labels/
#       test/
#         images/
#         labels/
#
# Structură de OUTPUT (ce vrem să obținem):
#
#   data/
#     images_enhanced/
#       train/
#         images/   <- JPG/PNG ENHANCED
#         labels/   <- TXT copiate 1:1
#       val/
#         images/
#         labels/
#       test/
#         images/
#         labels/
#
# Cod simplu, fără argparse, ușor de explicat la profesor.

import os
import cv2
import shutil

# =========================
# CONFIGURARE
# =========================

# TOTUL PLEACĂ DE AICI:
INPUT_ROOT = "data/images"             # aici ai train/val/test cu images+labels
OUTPUT_ROOT = "data/images_enhanced"   # aici vom crea noua structură

SPLITS = ["train", "val", "test"]
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".jfif")
LABEL_EXT = ".txt"


# =========================
# FUNCȚIE: ENHANCE IMAGINE
# =========================

def enhance_image(img):
    """
    Primește o imagine BGR și întoarce o variantă îmbunătățită.
    Pași:
      1. CLAHE pe canalul de luminanță (L) din LAB -> contrast mai bun.
      2. Unsharp mask -> contururi mai clare (zgârieturi, lovituri).
    """
    # BGR -> LAB
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # CLAHE: Contrast Limited Adaptive Histogram Equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)

    lab_enhanced = cv2.merge((l_enhanced, a, b))
    img_contrast = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    # Unsharp mask (sharpening)
    blurred = cv2.GaussianBlur(img_contrast, (0, 0), sigmaX=1.0, sigmaY=1.0)
    sharpen_amount = 1.0  # poți crește la 1.5 dacă vrei mai agresiv

    img_sharp = cv2.addWeighted(
        img_contrast,
        1.0 + sharpen_amount,
        blurred,
        -sharpen_amount,
        0
    )

    return img_sharp


# =========================
# IMAGINI: ENHANCE
# =========================

def process_images_split(split_name):
    """
    Pentru un split (train / val / test):
      - ia imaginile din data/images/<split>/images
      - aplică enhance_image
      - salvează în data/images_enhanced/<split>/images
    """
    input_dir = os.path.join(INPUT_ROOT, split_name, "images")
    output_dir = os.path.join(OUTPUT_ROOT, split_name, "images")

    if not os.path.exists(input_dir):
        print(f"[AVERTISMENT] Nu există dir imagini pentru '{split_name}': {input_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)

    print(f"[INFO] === Imagini {split_name} ===")
    print(f"       Intrare: {input_dir}")
    print(f"       Ieșire:  {output_dir}")

    num_gasite = 0
    num_salvate = 0

    for file_name in os.listdir(input_dir):
        lower = file_name.lower()
        if not lower.endswith(IMAGE_EXTS):
            continue  # nu e imagine

        num_gasite += 1

        input_path = os.path.join(input_dir, file_name)
        output_path = os.path.join(output_dir, file_name)

        img = cv2.imread(input_path)
        if img is None:
            print(f"[AVERTISMENT] Nu am putut citi: {input_path}")
            continue

        enhanced = enhance_image(img)

        ok = cv2.imwrite(output_path, enhanced)
        if ok:
            num_salvate += 1
        else:
            print(f"[AVERTISMENT] Nu am putut salva: {output_path}")

    print(f"[INFO] Imagini găsite: {num_gasite}, imagini salvate: {num_salvate}\n")


# =========================
# LABELS: COPIERE
# =========================

def copy_labels_split(split_name):
    """
    Pentru un split (train / val / test):
      - ia .txt din data/images/<split>/labels
      - le copiază în data/images_enhanced/<split>/labels
    """
    input_dir = os.path.join(INPUT_ROOT, split_name, "labels")
    output_dir = os.path.join(OUTPUT_ROOT, split_name, "labels")

    if not os.path.exists(input_dir):
        print(f"[AVERTISMENT] Nu există dir labels pentru '{split_name}': {input_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)

    print(f"[INFO] === Labels {split_name} ===")
    print(f"       Intrare: {input_dir}")
    print(f"       Ieșire:  {output_dir}")

    num_gasite = 0
    num_copiate = 0

    for file_name in os.listdir(input_dir):
        if not file_name.lower().endswith(LABEL_EXT):
            continue  # nu e .txt YOLO

        num_gasite += 1

        input_path = os.path.join(input_dir, file_name)
        output_path = os.path.join(output_dir, file_name)

        try:
            shutil.copy2(input_path, output_path)
            num_copiate += 1
        except Exception as e:
            print(f"[AVERTISMENT] Nu am putut copia {input_path} -> {output_path}: {e}")

    print(f"[INFO] Labels găsite: {num_gasite}, labels copiate: {num_copiate}\n")


# =========================
# MAIN
# =========================

if __name__ == "__main__":
    print("[INFO] Pornește scriptul ENHANCE FROM data/images")
    print(f"[INFO] INPUT_ROOT  = {INPUT_ROOT}")
    print(f"[INFO] OUTPUT_ROOT = {OUTPUT_ROOT}")
    print()

    for split in SPLITS:
        process_images_split(split)
        copy_labels_split(split)

    print("[INFO] Gata. Structura finală este în:", OUTPUT_ROOT)
    print("       (train/val/test cu images + labels)")
