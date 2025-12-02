# src/preprocessing/convert_coco_to_yolo.py
#
# scop: ia adnotările în format COCO din data/raw/cardd
#       și le transformă în format YOLO (cls x_c y_c w h normalizate),
#       păstrând structura train / validation / test pentru imagini + etichete.

import json
from pathlib import Path
from shutil import copy2
from collections import defaultdict
from tqdm import tqdm

# rădăcina proiectului (car-damage-detector/)
BASE_DIR = Path(__file__).resolve().parents[2]

# aici sunt json-urile COCO + imaginile brute
RAW_DIR = BASE_DIR / "data" / "raw" / "cardd"

# aici salvăm setul YOLO final (detecție, nu segmentare)
OUT_DIRS = {
    "train": BASE_DIR / "data" / "train",
    "val": BASE_DIR / "data" / "validation",
    "test": BASE_DIR / "data" / "test",
}

# subfolderele din RAW unde sunt imaginile
SPLIT_TO_IMG_SUBDIR = {
    "train": "train",
    "val": "val",
    "test": "test",
}


def coco_bbox_to_yolo(bbox, img_w, img_h):
    """
    COCO: [x_min, y_min, w, h]
    YOLO: x_c, y_c, w, h (toate normalizate la [0, 1] raportat la lățime/înălțime).
    """
    x, y, w, h = bbox
    x_c = (x + w / 2.0) / img_w
    y_c = (y + h / 2.0) / img_h
    w_n = w / img_w
    h_n = h / img_h
    return x_c, y_c, w_n, h_n


def convert_split(split: str):
    """convertește un singur split (train/val/test) din COCO în YOLO."""
    json_path = RAW_DIR / f"{split}.json"
    with open(json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    images = coco["images"]
    anns = coco["annotations"]
    cats = coco["categories"]

    # map image_id -> info imagine (nume + dimensiuni)
    img_info = {im["id"]: im for im in images}

    # grupăm adnotările pe imagine (image_id)
    anns_by_img = defaultdict(list)
    for a in anns:
        anns_by_img[a["image_id"]].append(a)

    # în COCO, id-urile claselor încep de la 1; YOLO se așteaptă la 0..N-1
    id_offset = min(c["id"] for c in cats)

    out_root = OUT_DIRS[split]
    img_out_dir = out_root / "images"
    lbl_out_dir = out_root / "labels"
    img_out_dir.mkdir(parents=True, exist_ok=True)
    lbl_out_dir.mkdir(parents=True, exist_ok=True)

    img_src_dir = RAW_DIR / SPLIT_TO_IMG_SUBDIR[split]

    print(f"\n=== Conversie split: {split.upper()} ===")
    for img in tqdm(images):
        img_id = img["id"]
        file_name = img["file_name"]
        w = img["width"]
        h = img["height"]

        # copiem imaginea (din raw în datasetul YOLO)
        src_path = img_src_dir / file_name
        dst_path = img_out_dir / file_name
        if not dst_path.exists():
            copy2(src_path, dst_path)

        # construim liniile YOLO (una per bounding box)
        yolo_lines = []
        for a in anns_by_img.get(img_id, []):
            cls = a["category_id"] - id_offset  # transformăm id-ul COCO în index 0..5
            x_c, y_c, w_n, h_n = coco_bbox_to_yolo(a["bbox"], w, h)
            yolo_lines.append(f"{cls} {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}")

        # scriem fișierul .txt aferent imaginii
        label_path = lbl_out_dir / (Path(file_name).stem + ".txt")
        with open(label_path, "w", encoding="utf-8") as f:
            f.write("\n".join(yolo_lines))

    print(f"Imagini YOLO salvate în: {img_out_dir}")
    print(f"Etichete YOLO salvate în: {lbl_out_dir}")


def main():
    for split in ["train", "val", "test"]:
        convert_split(split)


if __name__ == "__main__":
    main()
