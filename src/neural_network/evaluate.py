# src/neural_network/repair_and_evaluate.py
#
# 1) RESTORE test din quarantine_test (dacă există)
# 2) Verifică formatul labelurilor pe train/val/test (detect-only vs seg)
# 3) Dacă există seg labels reale -> rulează model.val()
# 4) Dacă NU există seg labels reale -> rulează model.predict() și salvează imagini cu predicții
#
# Rulare:
#   py src/neural_network/repair_and_evaluate.py

import os
import glob
import shutil
import json
from datetime import datetime
from ultralytics import YOLO

MODEL_PATH = "models/optimized_model.pt"
DATA_CONFIG = "config/car_damage.yaml"

IMPROVED_OUT_DIR = "results/predictions_val"  # unde salvăm imagini cu predicții dacă nu avem labels seg
OUT_JSON = "results/test_metrics.json"

ROOT_TRAIN = os.path.join("data", "images_enhanced", "train")
ROOT_VAL   = os.path.join("data", "images_enhanced", "val")
ROOT_TEST  = os.path.join("data", "images_enhanced", "test")

QUAR_DIR = os.path.join("data", "quarantine_test")

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def remove_cache_files():
    removed = 0
    for p in glob.glob(os.path.join("data", "**", "labels.cache"), recursive=True):
        try:
            os.remove(p)
            removed += 1
        except:
            pass
    return removed


def find_images_and_labels(root):
    images_dir = os.path.join(root, "images")
    labels_dir = os.path.join(root, "labels")
    if os.path.isdir(images_dir) and os.path.isdir(labels_dir):
        return images_dir, labels_dir
    raise FileNotFoundError(f"Nu găsesc structura standard: {root}/images și {root}/labels")


def count_images(images_dir):
    total = 0
    for ext in IMG_EXTS:
        total += len(glob.glob(os.path.join(images_dir, f"*{ext}")))
    return total


def token_count_first_object(label_path):
    # întoarce nr token-uri de pe prima linie non-goală
    try:
        with open(label_path, "r", encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                return len(line.split())
    except:
        return None
    return None


def analyze_split(root, name):
    images_dir, labels_dir = find_images_and_labels(root)
    n_imgs = count_images(images_dir)
    label_files = sorted(glob.glob(os.path.join(labels_dir, "*.txt")))

    counts = []
    examples_5 = []
    examples_6 = []
    examples_bad = []

    for p in label_files[:2000]:  # safe cap
        c = token_count_first_object(p)
        if c is None:
            examples_bad.append(p)
            continue
        counts.append(c)
        if c == 5 and len(examples_5) < 5:
            examples_5.append(p)
        if c >= 6 and len(examples_6) < 5:
            examples_6.append(p)

    detect_only = sum(1 for c in counts if c == 5)
    seg_like    = sum(1 for c in counts if c >= 6)

    report = {
        "split": name,
        "images_dir": images_dir,
        "labels_dir": labels_dir,
        "n_images": n_imgs,
        "n_labels": len(label_files),
        "detect_only_labels_est": detect_only,
        "seg_labels_est": seg_like,
        "bad_read_labels_est": len(examples_bad),
        "examples_detect_only": examples_5,
        "examples_seg": examples_6,
        "examples_bad": examples_bad[:5],
    }
    return report


def restore_test_from_quarantine():
    # dacă există quarantine_test/images -> mută înapoi în test/images
    q_img = os.path.join(QUAR_DIR, "images")
    q_lbl = os.path.join(QUAR_DIR, "labels")
    if not (os.path.isdir(q_img) and os.path.isdir(q_lbl)):
        return {"restored": False, "reason": "no quarantine_test/images+labels"}

    test_img = os.path.join(ROOT_TEST, "images")
    test_lbl = os.path.join(ROOT_TEST, "labels")
    os.makedirs(test_img, exist_ok=True)
    os.makedirs(test_lbl, exist_ok=True)

    moved_imgs = 0
    moved_lbls = 0

    for p in glob.glob(os.path.join(q_img, "*")):
        shutil.move(p, os.path.join(test_img, os.path.basename(p)))
        moved_imgs += 1

    for p in glob.glob(os.path.join(q_lbl, "*.txt")):
        shutil.move(p, os.path.join(test_lbl, os.path.basename(p)))
        moved_lbls += 1

    return {"restored": True, "moved_images": moved_imgs, "moved_labels": moved_lbls}


def safe_div(a, b):
    return a / b if b != 0 else 0.0


def extract_metrics(r):
    box = getattr(r, "box", None)
    seg = getattr(r, "seg", None)

    if box is None:
        raise RuntimeError("Nu am r.box. Verifică Ultralytics.")

    box_P = float(getattr(box, "mp", 0.0))
    box_R = float(getattr(box, "mr", 0.0))
    box_mAP50 = float(getattr(box, "map50", 0.0))
    box_mAP5095 = float(getattr(box, "map", 0.0))
    box_F1 = safe_div(2 * box_P * box_R, (box_P + box_R))

    payload = {
        "box_precision_mean": box_P,
        "box_recall_mean": box_R,
        "box_f1_mean": box_F1,
        "box_map50": box_mAP50,
        "box_map50_95": box_mAP5095,
    }

    if seg is not None:
        mask_P = float(getattr(seg, "mp", 0.0))
        mask_R = float(getattr(seg, "mr", 0.0))
        mask_mAP50 = float(getattr(seg, "map50", 0.0))
        mask_mAP5095 = float(getattr(seg, "map", 0.0))
        mask_F1 = safe_div(2 * mask_P * mask_R, (mask_P + mask_R))

        payload.update({
            "mask_precision_mean": mask_P,
            "mask_recall_mean": mask_R,
            "mask_f1_mean": mask_F1,
            "mask_map50": mask_mAP50,
            "mask_map50_95": mask_mAP5095,
        })

    return payload


def main():
    os.makedirs("results", exist_ok=True)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Nu există modelul: {MODEL_PATH}")
    if not os.path.exists(DATA_CONFIG):
        raise FileNotFoundError(f"Nu există config-ul dataset: {DATA_CONFIG}")

    print("[STEP A] Restore TEST from quarantine (if needed)...")
    restore_info = restore_test_from_quarantine()
    print("[INFO]", restore_info)

    print("[STEP B] Remove cache files...")
    print("[OK] cache removed:", remove_cache_files())

    print("[STEP C] Analyze splits...")
    rep_train = analyze_split(ROOT_TRAIN, "train")
    rep_val   = analyze_split(ROOT_VAL, "val")
    rep_test  = analyze_split(ROOT_TEST, "test")

    print("\n=== SPLIT REPORT ===")
    for rep in (rep_train, rep_val, rep_test):
        print(f"\n[{rep['split']}] images={rep['n_images']} labels={rep['n_labels']}")
        print(f"  detect_only_est={rep['detect_only_labels_est']}  seg_est={rep['seg_labels_est']}  bad_read_est={rep['bad_read_labels_est']}")
        print("  examples_detect_only:", rep["examples_detect_only"][:2])
        print("  examples_seg:", rep["examples_seg"][:2])

    # decidem unde putem evalua "real" seg
    # condiție minimă: seg_labels_est > 0 și imagini > 0
    candidates = []
    for rep in (rep_test, rep_val, rep_train):
        if rep["n_images"] > 0 and rep["seg_labels_est"] > 0:
            candidates.append(rep["split"])

    model = YOLO(MODEL_PATH)

    payload = {
        "evaluated_model": MODEL_PATH,
        "data_config": DATA_CONFIG,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "split_reports": {"train": rep_train, "val": rep_val, "test": rep_test},
    }

    if candidates:
        # preferăm test -> val
        split = "test" if "test" in candidates else ("val" if "val" in candidates else candidates[0])
        print(f"\n[STEP D] Running REAL evaluation on split='{split}' (seg labels detected)...")
        r = model.val(data=DATA_CONFIG, split=split, cache=False)
        payload["used_split"] = split
        payload["mode"] = "val_metrics"
        payload.update(extract_metrics(r))

        with open(OUT_JSON, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        print("[OK] Saved metrics to:", OUT_JSON)
        print("[DONE]")

    else:
        # NU există seg labels -> nu are sens să rulăm val() (va crăpa).
        # Facem predict pe val images și salvăm output.
        print("\n[WARN] NU am găsit niciun split cu seg labels (poligoane).")
        print("       Asta înseamnă că label-urile tale sunt detect-only (5 token-uri) sau invalide pentru seg.")
        print("       Nu putem calcula metrici seg. Fac predict pe VAL și salvez imaginile cu predicții.")

        # predict pe val images
        val_images_dir, _ = find_images_and_labels(ROOT_VAL)
        os.makedirs(IMPROVED_OUT_DIR, exist_ok=True)

        results = model.predict(
            source=val_images_dir,
            save=True,
            project=IMPROVED_OUT_DIR,
            name="run1",
            conf=0.25
        )

        payload["mode"] = "predict_only"
        payload["used_split"] = "val"
        payload["predict_output_dir"] = os.path.join(IMPROVED_OUT_DIR, "run1")

        with open(OUT_JSON, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        print("[OK] Saved report to:", OUT_JSON)
        print("[OK] Predictions saved to:", payload["predict_output_dir"])
        print("[DONE]")


if __name__ == "__main__":
    main()