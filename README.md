<div align="center">

<h1>🚗 Car Damage Detector</h1>

<h3>YOLO11m + CarDD • Deep Learning • UPB – FIIR</h3>

<br>

<img src="https://dummyimage.com/1200x260/111827/4c7dff&text=Car+Damage+Detector+AI" width="100%" style="border-radius:16px;">

<br><br>

<img src="https://img.shields.io/badge/Framework-YOLO11m-blue?style=for-the-badge">
<img src="https://img.shields.io/badge/Dataset-CarDD-green?style=for-the-badge">
<img src="https://img.shields.io/badge/Task-Object%20Detection-orange?style=for-the-badge">
<img src="https://img.shields.io/badge/WebUI-Flask-purple?style=for-the-badge">

</div>

---

# 🧩 1. Descrierea Proiectului

<div style="padding:18px; background:#0f172a; border-radius:12px; border:1px solid #1e293b;">

Acest proiect implementează un sistem inteligent de **detecție automată a daunelor auto** folosind rețele neuronale de tipul **YOLO11m**, antrenat pe dataset-ul Car Damage Dataset (**CarDD**).

Modelul poate detecta:

* zgârieturi (`scratch`)
* îndoiri (`dent`)
* crăpături (`crack`)
* geam spart (`glass_shatter`)
* far/stop spart (`lamp_broken`)
* pană la roată (`tire_flat`)

Aplicația include și o **interfață web** modernă care permite încărcarea unei fotografii și afișarea rezultatului YOLO într-un mod vizual și intuitiv.

</div>

---

# 🗂️ 2. Structura Proiectului

```
car-damage-detector/
│
├── README.md
├── config/
│   └── cardd_yolo.yaml
│
├── data/
│   ├── raw/cardd/        # dataset original COCO
│   ├── train/            # YOLO images + labels
│   ├── validation/
│   └── test/
│
├── models/
│   └── yolo11-cardd.pt   # modelul final antrenat
│
├── src/
│   ├── preprocessing/
│   │   ├── explore_cardd.py
│   │   └── convert_coco_to_yolo.py
│   ├── neural_network/
│   │   └── train_yolo11.py
│   └── web/
│       ├── app.py
│       ├── templates/
│       │   └── index.html
│       └── static/
│           ├── style.css
│           ├── uploads/
│           └── results/
│
└── runs/
```

---

# 📊 3. Analiza Datasetului CarDD

<div style="background:#0f172a; padding:20px; border-radius:12px; border:1px solid #1e293b">

### ✔ Statistici generale

* Train: **2816 imagini**
* Validation: **810 imagini**
* Test: **374 imagini**
* Total clase: **6**

### ✔ Clase disponibile

| ID | Clasă         |
| -- | ------------- |
| 0  | dent          |
| 1  | scratch       |
| 2  | crack         |
| 3  | glass_shatter |
| 4  | lamp_broken   |
| 5  | tire_flat     |

### ✔ Distribuția pe clase (train)

* scratch: **2560**
* dent: **1806**
* crack: **651**
* glass_shatter: **475**
* lamp_broken: **494**
* tire_flat: **225**

### ✔ Bounding box statistics

* bbox mediu: **78.608 px²**
* mediană: **29.108 px²**
* min: 60 px², max: 801453 px²
* adnotări / imagine: **2.21**

</div>

---

# 🔧 4. Preprocesarea Datasetului (COCO → YOLO)

<div style="padding:18px; background:#1e1b4b; border-left:5px solid #4f46e5; border-radius:12px;">

Script utilizat:
`src/preprocessing/convert_coco_to_yolo.py`

Acțiuni realizate:

✔ conversie bounding box COCO → YOLO
✔ etichete normalizate (x_center, y_center, w, h)
✔ structurare foldere YOLO (train/val/test)
✔ copierea imaginilor în format compatibil

</div>

---

# 🧠 5. Antrenarea Modelului YOLO11m

### ✔ Script:

`src/neural_network/train_yolo11.py`

### ✔ Parametri:

* `epochs = 200`
* `imgsz = 768`
* `batch = 8`
* augmentări avansate:

  * hsv transform
  * scaling
  * random erasing
  * flip LR

### ✔ Rezultate finale:

* **mAP50:** 0.736
* **mAP50-95:** 0.592

---

# 🌐 6. Interfața Web (Flask)

<div align="center">
<img src="https://dummyimage.com/1000x450/0f172a/38bdf8&text=Car+Damage+Web+Interface" width="90%" style="border-radius:16px;">
</div>

Funcționalități:

✔ Upload imagine
✔ Bara de progres animată la procesare
✔ Dark/Light theme
✔ Afișare comparativă: original vs rezultat YOLO
✔ Rezultatul salvat în `static/results/`

---

# 🚀 7. Workflow Demo

<div style="display:flex; justify-content:space-between; gap:20px;">

<div style="flex:1; min-width:250px; background:#0f172a; padding:20px; border-radius:12px;">
<h3>1️⃣ Upload</h3>
Imaginea se salvează automat în `/uploads`
</div>

<div style="flex:1; min-width:250px; background:#0f172a; padding:20px; border-radius:12px;">
<h3>2️⃣ Inferență YOLO</h3>
Modelul returnează imaginea cu bounding box-uri
</div>

<div style="flex:1; min-width:250px; background:#0f172a; padding:20px; border-radius:12px;">
<h3>3️⃣ Afișare rezultat</h3>
Interfața arată comparativ input ↔ output
</div>

</div>

---

# 📝 8. Concluzii

Acest proiect demonstrează:

* utilizarea unui dataset real (CarDD)
* preprocesare corectă COCO → YOLO
* antrenare YOLO11m de la zero
* implementare completă Web UI
* un sistem real de **constatare automată a daunelor auto**

---

# 👤 9. Autori

* **Baba Cristian-Teodor** – Student FIIR, UPB

---

# 🔥 Vrei să fac și:

✔ README pentru Etapa 4
✔ README pentru tot proiectul (landing page complet)
✔ Poster A3 pentru prezentare
✔ PDF final academic formatat perfect pentru profesor?
