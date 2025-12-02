# src/preprocessing/explore_cardd.py
#
# scop: să înțelegem datasetul CardDD în format COCO:
#       - câte imagini are fiecare split (train/val/test)
#       - câte adnotări și câte clase
#       - distribuția pe clase
#       - statistici de arie pentru bounding box-uri
#       - câte bbox-uri avem în medie pe imagine

import json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

# ne poziționăm la rădăcina proiectului (car-damage-detector/)
BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = BASE_DIR / "data" / "raw" / "cardd"

SPLITS = ["train", "val", "test"]


def load_coco(split: str) -> dict:
    """citește fișierul COCO (json) pentru un split (train/val/test)."""
    json_path = RAW_DIR / f"{split}.json"
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    for split in SPLITS:
        coco = load_coco(split)
        images = coco.get("images", [])
        anns = coco.get("annotations", [])
        cats = coco.get("categories", [])

        print(f"\n=== Split: {split.upper()} ===")
        print(f"Număr imagini: {len(images)}")
        print(f"Număr adnotări: {len(anns)}")
        print(f"Număr clase: {len(cats)}")

        # map id -> nume de clasă
        id2name = {c["id"]: c["name"] for c in cats}

        # distribuție pe clase (câte bbox-uri are fiecare clasă)
        cnt_cls = Counter(a["category_id"] for a in anns)
        print("Distribuție pe clase:")
        for cid, num in cnt_cls.items():
            name = id2name.get(cid, f"cls_{cid}")
            print(f"  - {name} (id={cid}): {num}")

        # statistici pe aria bbox-urilor
        areas = [a.get("area", 0.0) for a in anns]
        if areas:
            areas = np.array(areas, dtype=float)
            print("Statistici arie bounding box:")
            print(f"  - medie: {areas.mean():.2f}")
            print(f"  - mediană: {np.median(areas):.2f}")
            print(f"  - min: {areas.min():.2f}, max: {areas.max():.2f}")

        # câte adnotări avem pe fiecare imagine (bucăți de damage / imagine)
        anns_per_img = defaultdict(int)
        for a in anns:
            anns_per_img[a["image_id"]] += 1

        if anns_per_img:
            counts = np.array(list(anns_per_img.values()))
            print("Număr adnotări / imagine:")
            print(f"  - medie: {counts.mean():.2f}")
            print(f"  - min: {counts.min()}, max: {counts.max()}")


if __name__ == "__main__":
    main()
