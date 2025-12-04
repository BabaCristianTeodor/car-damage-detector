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

Fluxul sistemului este complet automat:

- încărcarea imaginii
- aplicarea de **filtre ENHANCE** (contrast, claritate, denoise)
- validarea imaginii
- inferență YOLO11m
- generare bounding box-uri sau mesaj „No damage detected”
- export imagine procesată
- logging pentru fiecare analiză

Interfața web este modernă și responsivă.

</div>

---

# 🗂️ 2. Structura Proiectului

```text
car-damage-detector/
│
├── README.md
├── config/
│   └── cardd_yolo.yaml
│
├── data/
│   ├── raw/cardd/
│   ├── train/
│   ├── validation/
│   └── test/
│
├── models/
│   └── yolo11-cardd.pt
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
````

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

Script utilizat: `src/preprocessing/convert_coco_to_yolo.py`

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

Funcționalități:

✔ Upload imagine
✔ Filtre ENHANCE automate înainte de analiză
✔ Bară de progres animată
✔ Dark/Light theme
✔ Afișare comparativă: input ↔ YOLO output
✔ Rezultatul salvat în `static/results/`

---

# 🚀 7. Workflow Demo

<div style="display:flex; justify-content:space-between; gap:20px;">

<div style="flex:1; min-width:250px; background:#0f172a; padding:20px; border-radius:12px;">
<h3>1️⃣ Upload</h3>
Imaginea intră automat în modul ENHANCE.
</div>

<div style="flex:1; min-width:250px; background:#0f172a; padding:20px; border-radius:12px;">
<h3>2️⃣ Enhance → Validare → YOLO</h3>
Filtre + validare + inferență YOLO11m.
</div>

<div style="flex:1; min-width:250px; background:#0f172a; padding:20px; border-radius:12px;">
<h3>3️⃣ Export rezultat</h3>
Bounding box-uri sau mesaj „No damage detected”.
</div>

</div>

---

# 🔄 8. Diagrama State Machine (Versiunea Finală)

## 8.1. Diagrama grafică

Imaginea diagramei este salvată în `docs/state_machine_car_damage.png`:

```markdown
![State Machine – Car Damage Detector](docs/state_machine_car_damage.png)
```

> Asigură-te că fișierul există în repo pe calea `docs/state_machine_car_damage.png`.

---

## 8.2. Descrierea stărilor și tranzițiilor

### 🔹 IDLE – „Așteaptă încărcare imagine de la utilizator”

* Stare de repaus a aplicației.
* Serverul Flask rulează, dar nu procesează nimic.
* Așteaptă ca utilizatorul să facă upload la o imagine cu mașina (avariată sau nu).

---

### 🔹 ENHANCE_IMAGE – „Aplică filtre de enhance: contrast / claritate / denoise”

* După upload, imaginea trece printr-un modul de **preprocesare**.
* Se aplică:

  * creștere de contrast (pentru a scoate în evidență zgârieturi și muchii),
  * claritate (sharpen),
  * denoise (pentru poze cu zgomot / lumină slabă).
* Scop: să fie mai ușor atât pentru YOLO, cât și pentru utilizator, să observe daunele.

---

### 🔹 VALIDATE_IMAGE – „Verifică format, rezoluție și dimensiune minimă”

* Verifică dacă fișierul:

  * este o imagine validă (JPG/PNG),
  * nu este corupt sau gol,
  * are rezoluție minimă acceptată pentru inferență.
* Dacă verificarea eșuează → tranziție către starea **ERROR**.
* Dacă totul este ok → tranziție către **PROCESS_IMAGE**.

---

### 🔹 PROCESS_IMAGE – „Rulează inferența YOLO – `has_defect = model(img)`”

* Imaginea preprocesată și validată este trimisă către modelul **YOLO11m**.
* Modelul returnează:

  * bounding boxes,
  * clase detectate (scratch, dent, crack etc.),
  * confidence score pentru fiecare detecție.
* Ieșiri posibile:

  * inferență reușită → **EXPORT_RESULT**;
  * eroare GPU / model / timeout → **ERROR**.

---

### 🔹 EXPORT_RESULT – „Desenează bounding box-uri, salvează imaginea, afișează în UI”

* Dacă `has_defect = True`:

  * se desenează bounding box-uri colorate pe imagine,
  * se salvează imaginea rezultată în `static/results/`.
* Dacă **nu s-a detectat nicio daună**:

  * se afișează un mesaj de tip *„No damage detected”*.
* În ambele cazuri:

  * se salvează un log (nume fișier, clase, scoruri, timp de procesare),
  * se afișează rezultatul în interfața web,
  * apoi aplicația revine în **IDLE** (pregătită pentru o nouă imagine).

---

### 🔹 ERROR – „Afișează mesaj de eroare, salvează log incident”

* Gestionează situații precum:

  * fișier corupt / format neacceptat,
  * eroare de inferență (model, GPU, memorie etc.).
* Utilizatorul vede un mesaj clar în UI (ex. „Fișier invalid” sau „Eroare la model”).
* Se salvează un log de incident pentru debugging.
* Din această stare se poate reveni în **IDLE**, pentru a încerca o altă imagine.

---

### 🔹 STOP – „Oprire aplicație / eliberare resurse”

* Reprezintă oprirea controlată a sistemului:

  * se eliberează resursele (GPU, fișiere temporare etc.),
  * se oprește serverul Flask.
* În practică este declanșată de închiderea aplicației sau oprirea serverului.

---

### 🔹 Tranziții importante

* **IDLE → ENHANCE_IMAGE** – când utilizatorul încarcă o imagine.
* **ENHANCE_IMAGE → VALIDATE_IMAGE** – după filtrare și pregătirea imaginii.
* **VALIDATE_IMAGE → PROCESS_IMAGE** – doar dacă fișierul este valid.
* **VALIDATE_IMAGE → ERROR** – dacă fișierul este corupt / invalid.
* **PROCESS_IMAGE → EXPORT_RESULT** – inferență YOLO reușită.
* **PROCESS_IMAGE → ERROR** – eroare la model / GPU / timp de execuție.
* **EXPORT_RESULT → IDLE** – export finalizat, sistemul așteaptă o nouă imagine.
* **ERROR → IDLE** – utilizatorul poate încerca din nou cu o altă imagine.

Prin această structură, sistemul se comportă ca un **Sistem Ciber-Fizic simplificat**, cu un flux clar: *input → prelucrare → decizie → output*, plus gestiunea erorilor.

---

# 🏭 9. SAF – Tabel Nevoie Reală → Soluție CPS → Modul Software

| Nevoie reală                      | Cum o rezolvă sistemul        | Modul software        |
| --------------------------------- | ----------------------------- | --------------------- |
| Detectarea rapidă a daunelor auto | YOLO11m cu inferență < 1s     | YOLO Inference Engine |
| Vizibilitate mai bună a daunelor  | Filtre ENHANCE automate       | Preprocessing Module  |
| Procesare robustă a imaginilor    | Validare + filtrare imagine   | Validator + Enhancer  |
| Export + evidență a rezultatelor  | Imagine finală + log JSON/CSV | Exporter + Logger     |

---

# 📝 10. Concluzii

Acest proiect demonstrează:

* utilizarea YOLO11m pe un dataset real (CarDD);
* preprocesare avansată prin ENHANCE pentru imagini cu calitate variabilă;
* detecție rapidă și precisă a daunelor auto;
* interfață web complet funcțională;
* State Machine industrial pentru flux autonom;
* logging complet pentru fiecare caz analizat.

---

# 👤 11. Autori

* **Baba Cristian-Teodor** – Student FIIR, UPB

```

- numele e identic (case-sensitive).
```
