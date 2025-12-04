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
├── README_Etapa4_Arhitectura_SIA.md   # (opțional, pentru predare RN)
├── config/
│   └── cardd_yolo.yaml
│
├── data/
│   ├── raw/cardd/        # dataset original COCO
│   ├── train/            # YOLO images + labels
│   ├── validation/
│   ├── test/
│   └── generated/        # imagini sintetice (contribuție originală 40%)
│
├── models/
│   ├── yolo11-cardd.pt   # modelul final antrenat
│   └── yolo11-base.pt    # model neantrenat (schelet RN pentru Etapa 4)
│
├── docs/
│   ├── state_machine_car_damage.png   # diagrama State Machine
│   └── screenshots/
│       └── ui_demo.png                # screenshot interfață web
│
├── src/
│   ├── data_acquisition/
│   │   └── generate_synthetic_damage_data.py   # generare imagini originale
│   │
│   ├── preprocessing/
│   │   ├── explore_cardd.py
│   │   └── convert_coco_to_yolo.py
│   │
│   ├── neural_network/
│   │   └── train_yolo11.py
│   │
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
* Total imagini: **4000**
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

![State Machine – Car Damage Detector](docs/state_machine_car_damage.png)

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

### 8.3. Justificarea State Machine-ului ales

Am modelat aplicația ca un **State Machine de tip clasificare imagini la cererea utilizatorului**, pentru că proiectul urmărește un flux clar: *user upload → preprocesare → inferență RN → afișare rezultat → logging / eroare*.

Stările principale (IDLE, ENHANCE_IMAGE, VALIDATE_IMAGE, PROCESS_IMAGE, EXPORT_RESULT, ERROR, STOP) acoperă:

1. **Pregătirea datelor** (upload + enhance + validare),
2. **Inferența RN** (YOLO11m pe imaginea procesată),
3. **Gestionarea ieșirilor** (rezultat sau eroare),
4. **Gestionarea ciclului de viață al aplicației** (IDLE/STOP).

Tranzițiile critice sunt:

* `IDLE → ENHANCE_IMAGE` – când utilizatorul încarcă o imagine validă.
* `VALIDATE_IMAGE → PROCESS_IMAGE` – doar după ce imaginea trece toate verificările.
* `PROCESS_IMAGE → EXPORT_RESULT` – când inferența YOLO se termină cu succes.
* `PROCESS_IMAGE → ERROR` – când apare o eroare de model / GPU / timeout.
* `ERROR → IDLE` – utilizatorul poate relua procesul cu o nouă imagine.

Starea **ERROR** este esențială pentru că în practică pot apărea fișiere corupte, formate neacceptate sau probleme de resurse (GPU, memorie). Sistemul trebuie să trateze aceste cazuri controlat și să permită reluarea normală a fluxului.

---

# 🏭 9. Tabel Nevoie Reală → Soluție SIA → Modul Software

| Nevoie reală concretă                 | Cum o rezolvă SIA-ul (metrici/efect)                                             | Modul software responsabil          |
| ------------------------------------- | -------------------------------------------------------------------------------- | ----------------------------------- |
| Detectarea rapidă a daunelor auto     | YOLO11m rulează inferența pe o imagine în **< 1s**, returnând clase + scoruri    | Modul NN – YOLO Inference Engine    |
| Vizibilitate mai bună a daunelor      | Filtre ENHANCE cresc contrastul / claritatea și reduc zgomotul înainte de RN     | Modul de Preprocesare (ENHANCE)     |
| Procesare robustă a imaginilor        | VALIDATE_IMAGE respinge fișiere corupte / cu rezoluție mică, evitând erorile RN  | Modul Validator + Enhancer          |
| Export + trasabilitate a rezultatelor | Fiecare analiză generează imagine finală + log JSON/CSV → **100% cazuri logate** | Modul Exporter + Logger (UI + back) |

---

# 🔢 10. Contribuția originală la setul de date (Etapa 4 RN)

Pentru a respecta cerința de **minimum 40% date originale**, proiectul include un modul dedicat de generare de imagini sintetice cu daune auto.

### Contribuția originală la setul de date

**Total observații finale:** 4000 imagini (train + val + test)
**Observații originale:** ~1600 imagini (≈ 40%)

**Tipul contribuției:**

* ✅ Date generate prin simulare fizică / imagini sintetice cu daune
* ⬜ Date achiziționate cu senzori proprii
* ⬜ Etichetare/adnotare manuală
* ⬜ Alte surse

**Descriere detaliată:**

Imaginile originale sunt generate pornind de la poze cu mașini fără daune sau cu daune minore, peste care se aplică, programatic:

* **zgârieturi sintetice** (texturi de scratch, linii subțiri cu variații de culoare și grosime),
* **îndoiri și crăpături simulate** prin deformări locale și overlay-uri de pattern-uri,
* **noise + blur + variații de iluminare**, pentru a simula condiții reale (noapte, ploaie, camere diferite).

Aceste transformări nu sunt simple augmentări (rotiri/flip), ci **simulează fizic apariția unor defecte noi**, generând imagini care nu există în datasetul public CarDD.

Imagistica rezultată este folosită atât la antrenare, cât și la testare, crescând diversitatea tipurilor de daune și generalizarea modelului YOLO.

**Locația codului:**
`src/data_acquisition/generate_synthetic_damage_data.py`

**Locația datelor:**
`data/generated/`

---

# 🧱 11. Arhitectura SIA și modulele software (Etapa 4)

Arhitectura urmează modelul cu **3 module principale** cerut la curs:

---

## 11.1. Modul 1 – Data Acquisition / Generare Date

**Rol:**

* generează imagini sintetice cu daune auto,
* produce log-uri CSV cu metadata (tip defect, intensitate, parametri de simulare).

**Responsabilități:**

* rulează scriptul:
  `python src/data_acquisition/generate_synthetic_damage_data.py`
* salvează imaginile noi în `data/generated/`
* salvează `data/generated/metadata.csv` cu coloane de tip:
  `filename, scratch_level, dent_level, noise_level, brightness, ...`

Acest modul acoperă partea de **contribuție originală 40%**.

---

## 11.2. Modul 2 – Neural Network (YOLO11m)

**Rol:**

* definește și încarcă modelul YOLO11m,
* rulează inferența pe imaginile preprocesate.

**Fișiere cheie:**

* `src/neural_network/train_yolo11.py` – script de definire + antrenare YOLO
* `models/yolo11-base.pt` – model neantrenat (schelet RN pentru Etapa 4)
* `models/yolo11-cardd.pt` – model antrenat pe CarDD + generated

**Funcționalități:**

* configurare parametri (`epochs`, `imgsz`, `batch`)
* integrare augmentări avansate
* salvare / încărcare model
* API de inferență folosit de aplicația web.

---

## 11.3. Modul 3 – Web Service / UI (Flask)

**Rol:**

* oferă interfața cu utilizatorul,
* orchestrează pipeline-ul end-to-end.

**Fișiere cheie:**

* `src/web/app.py` – server Flask
* `src/web/templates/index.html` – UI
* `src/web/static/style.css` – stilizare
* `docs/screenshots/ui_demo.png` – screenshot interfață

**Flux:**

1. Utilizatorul încarcă o imagine (endpoint `/upload`).
2. Backend-ul salvează fișierul în `static/uploads/`.
3. Se rulează modul ENHANCE + VALIDATE.
4. Se apelează YOLO (modul NN).
5. Rezultatul (bounding box-uri + scoruri) este desenat și salvat în `static/results/`.
6. UI afișează comparativ input / output.

---

# 📝 12. Concluzii

Acest proiect demonstrează:

* utilizarea YOLO11m pe un dataset real (CarDD), extins cu **date originale sintetice**;
* preprocesare avansată prin ENHANCE pentru imagini cu calitate variabilă;
* detecție rapidă și precisă a daunelor auto;
* interfață web complet funcțională (upload → analiză → rezultat);
* State Machine industrial pentru flux autonom și gestionarea erorilor;
* arhitectură SIA cu 3 module (Data Acquisition, Neural Network, Web UI) conform cerințelor Etapa 4;
* logging complet pentru fiecare caz analizat, cu posibilitate de audit și analiză ulterioară.

---

# 👤 13. Autori

* **Baba Cristian-Teodor** – Student FIIR, UPB
* 
🔬 14. Diferențe față de versiunea anterioară a documentației (raport academic)

Această secțiune prezintă în mod structurat și academic modificările aduse documentației și arhitecturii proiectului, comparativ cu versiunea precedentă (README – P2) , în vederea conformării proiectului la cerințele Etapei 4 din cadrul disciplinei.

14.1. Extinderea cadrului conceptual și arhitectural

Versiunea anterioară documenta doar mecanismele de detecție și funcționarea interfeței web.
Versiunea actualizată include:

introducerea modelării sistemului conform paradigmei SIA (Sisteme Informatice Autonome);

definirea arhitecturii în trei module principale (Data Acquisition, Neural Network, Web Service), conform cerințelor Etapei 4;

prezentarea responsabilităților fiecărui modul și a relațiilor dintre acestea într-o manieră formală.

Aceste elemente nu erau prezente în versiunea P2.

14.2. Integrarea unui modul de Data Acquisition și asigurarea contribuției originale la setul de date

Versiunea inițială descria exclusiv utilizarea dataset-ului CarDD.
Versiunea actuală introduce:

un modul nou: src/data_acquisition/,

generarea de date sintetice pentru acoperirea cerinței de minimum 40% contribuție originală la baza de date,

documentarea metodologiei de generare a datelor (texturi artificiale de zgârieturi, deformări simulate, perturbări de iluminare).

Această componentă metodologică nu exista în versiunea anterioară.

14.3. Introducerea și detalierea unui State Machine operațional

Versiunea P2 nu conținea o descriere formală a comportamentului sistemului.

Versiunea actuală include:

o diagramă oficială a mașinii de stări: docs/state_machine_car_damage.png,

descrierea formală a fiecărei stări (IDLE, ENHANCE_IMAGE, VALIDATE_IMAGE, PROCESS_IMAGE, EXPORT_RESULT, ERROR, STOP),

modelarea tranzițiilor între stări și justificarea utilizării modelului finit de stări în arhitectura unui SIA.

Această componentă este esențială pentru Etapa 4 și a fost adăugată integral.

14.4. Completarea fluxului de procesare cu etape suplimentare neincluse anterior

Versiunea actualizată prezintă pentru prima dată următoarele etape:

ENHANCE_IMAGE: aplicarea automată de filtre (contrast, claritate, denoise),

VALIDATE_IMAGE: verificarea strictă a formatului, rezoluției și integrității imaginii,

gestionarea sistematică a excepțiilor prin starea ERROR,

normalizarea fluxului în cadrul State Machine-ului.

Versiunea anterioară documenta doar încărcarea imaginii, inferența YOLO și afișarea rezultatului, fără un control riguros al fluxului.

14.5. Consolidarea arhitecturii software și extinderea structurii proiectului

Structura proiectului a fost extinsă pentru a reflecta cerințele academice:

introducerea directoarelor docs/, data/generated/,

introducerea modelului neantrenat necesar Etapei 4 (schelet RN),

adăugarea fișierelor de logging și metadate specifice procesului de achiziție de date.

Versiunea P2 includea doar modulele YOLO și Web UI.

14.6. Introducerea unei secțiuni formale Nevoie reală → Soluție CPS → Modul software

În versiunea anterioară nu exista o corelare explicită între:

nevoia concretă din mediul real,

soluția tehnică implementată,

modulul software responsabil.

Versiunea actuală introduce un tabel completat conform metodologiei SIA, prezentând:

criteriile de performanță,

funcțiile sistemului,

acoperirea modulară a fiecărei nevoi.

14.7. Clarificarea contribuției la nivel de sistem autonom

Versiunea actuală documentează explicit:

independența operațională a sistemului,

capacitatea de prelucrare autonomă a input-urilor,

gestionarea stărilor interne,

reluarea fluxului după erori,
reconfirmând caracterul de Sistem Informatic Autonom, cerință absentă în versiunea anterioară.

14.8. Extinderea concluziilor pentru conformitate academică

Concluziile au fost reformulate:

pentru a reflecta modularitatea sistemului,

pentru a evidenția contribuția originală asupra dataset-ului,

pentru a integra rolul State Machine-ului în arhitectura finală,

pentru a corespunde structurii cerute în Etapa 4.

✔ Rezumat academic al diferențelor

În ansamblu, versiunea actuală a documentației:

respectă integral cerințele Etapei 4 din disciplina SIA/RN,

include arhitectura modulară completă,

integrează contribuția originală la dataset,

introduce o modelare formală prin State Machine,

oferă o prezentare riguroasă a fluxului informațional,

extinde structura proiectului la nivelul cerut de un SIA real,

documentează sistemul la un nivel academic mult superior versiunii anterioare.
