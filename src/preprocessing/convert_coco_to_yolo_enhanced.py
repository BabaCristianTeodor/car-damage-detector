# src/preprocessing/convert_coco_to_yolo_enhanced.py
#
# scop: la fel ca convert_coco_to_yolo.py, DAR:
#       - etichetele YOLO sunt generate tot din json-urile COCO originale
#       - imaginile copiate în datasetul YOLO sunt cele ENHANCED
#         (data/processed/cardd_enhanced/<split>/images)

import json
from pathlib import Path
from shutil import copy2
from collections import defaultdict
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parents[2]

# json-urile COCO (rămân aceleași, nu le modificăm)
RAW_DIR = BASE_DIR / "data" / "raw" / "cardd"

# imaginile preprocesate (resize + enhance)
ENH_ROOT = BASE_DIR / "data" / "processed" / "cardd_enhanced"

# aici va fi noul set YOLO bazat pe imagini enhanced
OUT_DIRS = {
    "train": BASE_DIR / "data" / "train_enh",
    "val": BASE_DIR / "data" / "validation_enh",
    "test": BASE_DIR / "data" / "test_enh",
}


def coco_bbox_to_yolo(bbox, img_w, img_h):
    """COCO [x, y, w, h] -> YOLO x_c, y_c, w, h normalizate."""
    x, y, w, h = bbox
    x_c = (x + w / 2.0) / img_w
    y_c = (y + h / 2.0) / img_h
    w_n = w / img_w
    h_n = h / img_h
    return x_c, y_c, w_n, h_n


def convert_split(split: str):
    """
    convertește un split (train/val/test):
      - citește json-ul COCO original
      - copiază imaginile ENHANCED
      - generează etichetele YOLO în funcție de bbox-urile din COCO
    """
    json_path = RAW_DIR / f"{split}.json"
    with open(json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    images = coco["images"]
    anns = coco["annotations"]
    cats = coco["categories"]

    # map image_id -> info imagine
    img_info = {im["id"]: im for im in images}

    # grupăm adnotările pe imagine
    anns_by_img = defaultdict(list)
    for a in anns:
        anns_by_img[a["image_id"]].append(a)

    # id-urile claselor COCO încep de la 1; YOLO vrea 0..N-1
    id_offset = min(c["id"] for c in cats)

    out_root = OUT_DIRS[split]
    img_out_dir = out_root / "images"
    lbl_out_dir = out_root / "labels"
    img_out_dir.mkdir(parents=True, exist_ok=True)
    lbl_out_dir.mkdir(parents=True, exist_ok=True)

    # imaginile enhanced pentru acest split
    enh_img_dir = ENH_ROOT / split / "images"

    print(f"\n=== Conversie ENHANCED split: {split.upper()} ===")
    for img in tqdm(images):
        img_id = img["id"]
        file_name = img["file_name"]
        w = img["width"]
        h = img["height"]

        # sursa: imaginea enhanced (același nume de fișier)
        src_path = enh_img_dir / file_name
        dst_path = img_out_dir / file_name

        if not src_path.exists():
            # dacă lipsește enhanced, poți decide să folosești fallback la raw sau doar warning
            print(f"[WARN] Imagine enhanced lipsă: {src_path}")
            continue

        if not dst_path.exists():
            copy2(src_path, dst_path)

        # generăm liniile YOLO pentru toate bbox-urile din imagine
        yolo_lines = []
        for a in anns_by_img.get(img_id, []):
            cls = a["category_id"] - id_offset
            x_c, y_c, w_n, h_n = coco_bbox_to_yolo(a["bbox"], w, h)
            yolo_lines.append(f"{cls} {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}")

        label_path = lbl_out_dir / (Path(file_name).stem + ".txt")
        with open(label_path, "w", encoding="utf-8") as f:
            f.write("\n".join(yolo_lines))

    print(f"Imagini YOLO ENH salvate în: {img_out_dir}")
    print(f"Etichete YOLO ENH salvate în: {lbl_out_dir}")


def main():
    for split in ["train", "val", "test"]:
        convert_split(split)


if __name__ == "__main__":
    main()
