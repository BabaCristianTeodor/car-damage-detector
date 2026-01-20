---
<div align="center">

# 🚗 **CAR DAMAGE DETECTION SYSTEM**

## *Sistem inteligent pentru detecția daunelor auto folosind Rețele Neuronale*
</div>

<p align="center">
  <img src="https://img.shields.io/badge/Neural%20Networks-UPB%20FIIR-blueviolet">
  <img src="https://img.shields.io/badge/YOLO-v11m-0A66C2">
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C">
  <img src="https://img.shields.io/badge/GPU-RTX%204060-success">
  <img src="https://img.shields.io/badge/Project-Academic%20Final-brightgreen">
</p>

<p align="center">
  <b>Universitatea POLITEHNICA București</b><br>
  Facultatea de Inginerie Industrială și Robotică (FIIR)<br>
  Disciplina: <b>Rețele Neuronale</b><br><br>
  Student: <b>Baba Cristian Teodor</b><br>
  An universitar: 2025–2026
</p>

---

## 🌌 Introducere

Proiectul **Car Damage Detection System** reprezintă implementarea unui **sistem inteligent complet**, bazat pe **rețele neuronale convoluționale**, capabil să detecteze automat daunele vizibile ale unui vehicul dintr-o imagine digitală.

Sistemul nu se limitează la antrenarea unui model, ci acoperă **întregul ciclu de viață al unei aplicații bazate pe RN**:

* analiză și pregătire dataset,
* definirea arhitecturii software,
* antrenare și evaluare,
* inferență reală într-o aplicație funcțională,
* optimizare și selecția modelului final.

---

## 🧠 Tipul problemei abordate

<table>
<tr><td><b>Tip problemă</b></td><td>Object Detection (Computer Vision)</td></tr>
<tr><td><b>Date de intrare</b></td><td>Imagini RGB cu vehicule avariate</td></tr>
<tr><td><b>Date de ieșire</b></td><td>Bounding box-uri + clasă + scor de încredere</td></tr>
<tr><td><b>Model RN</b></td><td>YOLO11m</td></tr>
</table>

---

## 🏷️ Clase de daune detectate

<p align="center">
  <img src="https://img.shields.io/badge/dent-gray">
  <img src="https://img.shields.io/badge/scratch-blue">
  <img src="https://img.shields.io/badge/crack-purple">
  <img src="https://img.shields.io/badge/glass_shatter-cyan">
  <img src="https://img.shields.io/badge/lamp_broken-orange">
  <img src="https://img.shields.io/badge/tire_flat-red">
</p>

Aceste clase au fost alese pentru a acoperi atât:

* **defecte structurale** (dent, crack),
* cât și **defecte funcționale** (lamp_broken, tire_flat).

---

# 📂 Structura actuală a proiectului

Structura reflectă implementarea reală și separarea clară a responsabilităților:

project-root/
├── README.md
├── requirements.txt
├── startweb.txt                  
│
├── config/
│   └── car_damage.yaml
│
├── data/
│   ├── images/
│   │   ├── manual/               
│   │   │   ├── images/
│   │   │   └── labels/
│   │   ├── train/
│   │   │   ├── images/
│   │   │   └── labels/
│   │   ├── val/
│   │   │   ├── images/
│   │   │   └── labels/
│   │   └── test/
│   │       ├── images/
│   │       └── labels/
│   │
│   └── images_enhanced/
│       ├── train/
│       │   ├── images/
│       │   └── labels/  
│       ├── val/
│       │   ├── images/
│       │   └── labels/ 
│       └── test/
│           ├── images/
│           └── labels/
│
├── models/
│   └── optimized_model.pt
│
├── results/
│   ├── training_history.csv
│   ├── test_metrics.json
│   ├── final_metrics.json
│   ├── optimization_experiments.csv
│   └── hyperparameters.yaml
│
├── runs/
│   ├── rn_train/               
│   ├── detect/              
│   └── opt/    
│
├── docs/
│   ├── loss_curve.png
│   ├── confusion_matrix_normalized.png
│   └── screenshots/
│       ├── inference_real.png
│       └── inference_optimized_f.png
│
└── src/
    ├── preprocessing/
    │   └── enhance_images.py
    │
    ├── neural_network/
    │   ├── train.py
    │   ├── evaluate.py
    │   └── plot_loss_curve.py
    │
    └── app/
        └── main.py

---

# 🧪 **ETAPA 3 — Analiza și pregătirea setului de date**

Această etapă a avut rolul de a asigura **calitatea datelor**, fără a introduce artificii inutile care ar putea distorsiona generalizarea.

### Aspecte cheie:

* dataset organizat strict în format YOLO;
* separare clară train / validation / test;
* etichete verificate manual;
* fără augmentări agresive.

### Preprocesare utilizată

✔ **ENHANCE (soft)**  
✔ evidențiere defecte fine  
✔ păstrarea fidelității imaginii originale  

Această abordare minimizează riscul de **overfitting artificial** și păstrează un pipeline ușor de justificat academic.

---

## 🏗️ ETAPA 4 — Arhitectura sistemului (State Machine)

Aplicația este modelată ca o **mașină de stări (State Machine)**, oferind control complet asupra fluxului de execuție și o integrare clară a rețelei neuronale într-un sistem software real.

### Principii arhitecturale

* determinism al execuției;
* tratare explicită a erorilor;
* reset controlat al aplicației;
* separare clară între etapele logice ale pipeline-ului RN.

### Diagrama de stări a aplicației

<p align="center">
  <img src="docs/state_machine_car_damage.png" width="85%">
</p>

<p align="center">
  <i>Figura 1 – Diagrama State Machine a sistemului de detecție a daunelor auto</i>
</p>

Fiecare stare din diagramă corespunde unei faze logice distincte:

* încărcarea imaginii de către utilizator;
* aplicarea preprocesării ENHANCE;
* validarea formatului și dimensiunii imaginii;
* rularea inferenței YOLO;
* afișarea rezultatelor sau tratarea erorilor;
* resetarea aplicației sau oprirea controlată.

Această abordare permite o analiză clară a fluxului și o implementare robustă, ușor de extins.

---

# 🤖 ETAPA 5 — Antrenarea și evaluarea rețelei neuronale

### Configurație utilizată

| Componentă    | Specificație          |
| ------------- | --------------------- |
| GPU           | NVIDIA RTX 4060 – 8GB |
| Framework     | PyTorch + Ultralytics |
| Model         | YOLO11m               |
| Mod de rulare | Local                 |

### Procesul de învățare

Pentru analiza comportamentului rețelei neuronale pe parcursul antrenării a fost monitorizată evoluția funcției de pierdere (loss) atât pe setul de antrenare, cât și pe setul de validare.

<p align="center">
  <img src="docs/loss_curve.png" width="85%">
</p>

<p align="center">
  <i>Figura 2 – Evoluția loss-ului de antrenare și validare</i>
</p>

Graficul *Loss vs Validation Loss* evidențiază:

* o scădere constantă a loss-ului de antrenare;
* stabilitatea loss-ului pe setul de validare;
* lipsa unui overfitting sever;
* activarea corectă a mecanismului de **early stopping**.

Acest comportament indică o convergență stabilă a modelului.

---

# 🖥️ Inferență reală – demonstrație (model antrenat)

Aplicația finală permite rularea inferenței YOLO pe imagini reale, utilizând același pipeline definit în etapele anterioare.

<p align="center">
  <img src="docs/screenshots/inference_real.png" width="90%">
</p>

<p align="center">
  <i>Figura 3 – Exemplu de inferență reală (Etapa 5): imagine originală, preprocesată și rezultatul YOLO</i>
</p>

Funcționalități demonstrate:

* încărcarea unei imagini reale;
* aplicarea automată a preprocesării ENHANCE;
* detectarea daunelor prin YOLO;
* afișarea bounding box-urilor și a scorurilor de încredere.

---

## ⚠️ Limitări și observații (context pentru interpretarea metricilor)

* confuzie între clase vizual similare (`scratch` vs `crack`);
* sensibilitate la variații de iluminare și reflexii ale caroseriei;
* distribuție neechilibrată a claselor în setul de date;
* **dimensiunea redusă a setului de date**, determinată de necesitatea **limitării numărului de imagini** pentru a asigura o **rulare eficientă a procesului de antrenare** și pentru a evita supraîncărcarea resurselor hardware disponibile (GPU și memorie);
* ca urmare a acestei reduceri, **clasele cu defecte fine** (`scratch` și `crack`) dispun de mai puține exemple relevante, ceea ce conduce la **performanțe mai scăzute de recunoaștere** comparativ cu defectele cu contrast vizual ridicat.

Această limitare este specifică scenariilor de antrenare pe resurse hardware locale și nu reprezintă o deficiență conceptuală a arhitecturii alese.

---

# 🧪 ETAPA 6 — Optimizare, selecția modelului și evaluare finală

În această etapă a fost realizată optimizarea performanței modelului YOLO utilizat pentru detecția daunelor auto, prin rularea mai multor experimente controlate și evaluarea comparativă a rezultatelor obținute.

Scopul principal a fost selecția unui model final cu performanță globală superioară, utilizând metrici standard din domeniul *object detection*.

---

## 📏 Metrică de evaluare utilizată

Pentru evaluarea și compararea experimentelor a fost utilizată ca metrică principală:

**mAP@50–95 (mean Average Precision pe multiple praguri IoU)**

Aceasta este considerată metrică standard în detecția de obiecte, oferind o evaluare robustă a calității localizării și clasificării.

Metricile Precision și Recall au fost analizate complementar, pentru interpretarea comportamentului modelului.

---

## 🔬 Experimente de optimizare (exp1–exp4)

Au fost realizate patru experimente de optimizare, denumite `exp1` – `exp4`. Pentru fiecare experiment a fost analizată performanța pe setul de validare.

Rezultatele complete sunt documentate în fișierul:

results/optimization_experiments.csv

yaml
Copy code

### 📊 Rezultate comparative (ultimul epoch)

| Experiment | mAP@50 | mAP@50–95 | Precision | Recall |
|-----------|--------|-----------|-----------|--------|
| exp1 | 0.49669 | 0.37809 | 0.58170 | 0.46708 |
| exp2 | 0.49669 | 0.37809 | 0.58170 | 0.46708 |
| exp3 | **0.51103** | **0.38338** | 0.56153 | **0.49470** |
| exp4 | 0.38352 | 0.27721 | 0.45227 | 0.43409 |

### Observații analitice asupra experimentelor

- **exp1 vs exp2**: rezultate identice (același comportament de convergență), sugerând că modificarea introdusă nu a avut impact măsurabil sau a fost neutralizată de setările implicite ale pipeline-ului YOLO.
- **exp3**: obține cel mai bun compromis global, crescând atât **mAP@50**, cât și **mAP@50–95**, concomitent cu îmbunătățirea recall-ului; acest lucru indică o generalizare mai robustă, nu doar o creștere punctuală a unei singure metrici.
- **exp4**: degradare semnificativă pe toate metricile, ceea ce sugerează fie o configurație instabilă, fie un regim de învățare nepotrivit pentru distribuția dataset-ului.

---

## 🏆 Selecția modelului optim

Pe baza valorii maxime obținute pentru **mAP@50–95**, experimentul **exp3** a fost ales ca model optimizat final.

**Motivație:** exp3 oferă cea mai bună performanță globală, având cel mai ridicat scor mAP@50–95 și un recall superior, indicând o capacitate mai bună de detectare a daunelor auto pe setul de validare.

---

## 💾 Modelul final utilizat

Modelul rezultat în urma Etapei 6 este salvat ca:

models/optimized_model.pt

yaml
Copy code

Acest model înlocuiește complet versiunea utilizată în Etapa 5 și reprezintă modelul final al proiectului.

---

## 📉 Confusion Matrix și evaluare finală

Confusion Matrix pentru modelul optimizat a fost generată în urma evaluării pe setul de validare.

<p align="center">
  <img src="docs/confusion_matrix_normalized.png" width="85%">
</p>

<p align="center">
  <i>Figura 4 – Matricea de confuzie a sistemului (model optimizat)</i>
</p>

Analiza evidențiază confuzii între clase vizual similare (ex. `scratch` și `crack`), precum și o performanță superioară pentru defectele cu contrast vizual ridicat.

---

## 🖥️ Integrarea în aplicația finală (model optimizat)

Aplicația UI a fost actualizată pentru a utiliza exclusiv modelul optimizat (`optimized_model.pt`), asigurând consistența între etapa de evaluare și inferența realizată în aplicația finală.

<p align="center">
  <img src="docs/screenshots/inference_optimized_f.png" width="85%">
</p>

<p align="center">
  <i>Figura 5 – Screenshot: inferență cu modelul optimizat încărcat și testat</i>
</p>

---

## 📊 Metrici finale (model optimizat)

Metricile finale obținute pentru modelul optimizat sunt salvate în:

results/final_metrics.json

yaml
Copy code

Valori raportate:

- Precision (macro): **0.568**
- Recall (macro): **0.495**
- mAP@50: **0.512**
- mAP@50–95: **0.383**

---

## 📐 Interpretare detaliată a coeficienților (metricilor) — secțiunea critică

În object detection, interpretarea metricilor trebuie făcută în context, deoarece fiecare coeficient descrie un aspect diferit al comportamentului modelului. În plus, pentru defecte auto (mai ales cele fine), localizarea exactă a conturului este intrinsec dificilă, ceea ce afectează direct scorurile mAP stricte.

### 🔹 Precision (Precizia)

Precision reprezintă proporția predicțiilor corecte din totalul predicțiilor făcute:

\[
Precision = \frac{TP}{TP + FP}
\]

**Precision ≈ 0.568** indică faptul că modelul produce, în majoritatea cazurilor, detecții valide (număr relativ redus de alarme false). În practică, aceasta înseamnă că sistemul este mai degrabă „conservator”: preferă să nu raporteze o daună decât să raporteze una incorect.

Acest comportament este dezirabil în aplicații de inspecție, deoarece minimizează situațiile în care utilizatorul este indus în eroare de detecții artificiale.

---

### 🔹 Recall (Rata de detecție)

Recall măsoară proporția daunelor reale detectate corect:

\[
Recall = \frac{TP}{TP + FN}
\]

**Recall ≈ 0.495** arată că o parte dintre daune nu sunt detectate, în special în cazul:
- defectelor subțiri, cu contur difuz (`scratch`);
- defectelor cu textură asemănătoare fundalului (`crack`);
- claselor slab reprezentate (dataset neechilibrat).

În termeni practici, recall-ul reflectă cât de „sensibil” este sistemul: un recall mai mare ar însemna mai puține ratări, dar de regulă cu riscul creșterii alarmelor false (scăderea precision). În acest proiect, echilibrul obținut este realist pentru un pipeline rulat local și un dataset limitat.

---

### 🔹 mAP@50 (Mean Average Precision la IoU 0.5)

mAP@50 este performanța la un prag IoU permisiv, unde bounding box-ul trebuie să se suprapună decent cu adevărul de referință, dar nu perfect.

**mAP@50 ≈ 0.512** indică faptul că modelul:
- identifică în mod corect defectele și zona aproximativă a acestora;
- se comportă consistent pe majoritatea scenariilor.

Pentru multe aplicații practice de triere/filtrare inițială, mAP@50 este suficient pentru a considera sistemul util.

---

### 🔹 mAP@50–95 (metrică principală, strictă)

mAP@50–95 este metrica cea mai exigentă: media performanței pe praguri IoU de la 0.50 la 0.95. Aceasta penalizează puternic localizările imprecise și bounding box-urile care nu conturează exact defectul.

**mAP@50–95 ≈ 0.383** este o valoare realistă pentru detecția de defecte auto, deoarece:
- defectele sunt adesea mici/alungite și greu de încadrat exact;
- anotările umane au variații inerente (conturul zgârieturilor nu este obiectiv);
- dataset-ul este neechilibrat (clasele rare scad media);
- nu s-au folosit augmentări agresive (decizie intenționată pentru realism și justificare academică).

Diferența dintre **mAP@50 (0.512)** și **mAP@50–95 (0.383)** este un indicator clar că modelul recunoaște defectele, dar localizarea foarte precisă rămâne partea cea mai dificilă.

---

### 🔎 Corelarea metricilor cu realitatea aplicației

Combinația obținută:
- Precision moderată spre ridicată,
- Recall moderat,
- mAP@50 solid,
- mAP@50–95 strict,

descrie un model echilibrat: oferă detecții valide și stabile, dar este constrâns de natura dataset-ului și de dificultatea intrinsecă a localizării defectelor fine.

Acesta este exact tipul de rezultat care ar trebui obținut într-un proiect academic realist, fără supra-optimizare artificială.

---

## ✅ Concluzie generală

Proiectul demonstrează implementarea completă a unui sistem de detecție a daunelor auto bazat pe rețele neuronale, incluzând:
- antrenare și evaluare riguroasă;
- optimizare prin experimente controlate și selecția justificată a modelului;
- integrarea într-o aplicație funcțională de inferență;
- interpretare academică a coeficienților de performanță.

Soluția rezultată reprezintă o aplicație practică solidă a rețelelor neuronale în domeniul *computer vision* și oferă o bază robustă pentru extindere.

---

## 🚀 Posibile direcții de dezvoltare

* extinderea dataset-ului și rebalansarea claselor;
* fine-tuning dedicat pentru defecte fine (`scratch`, `crack`);
* inferență pe secvențe video;
* analiză comparativă cu alte arhitecturi (YOLO variants / Faster R-CNN / RetinaNet).

---

> **Car Damage Detection System** reprezintă o aplicație practică solidă a rețelelor neuronale în domeniul computer vision.
