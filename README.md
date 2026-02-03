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
```
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
```
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

Aceasta este considerată metrica standard în detecția de obiecte, oferind o evaluare robustă a calității localizării și clasificării.

Metricile Precision și Recall au fost analizate complementar, pentru a înțelege echilibrul dintre:
- capacitatea modelului de a evita alarmele false (Precision),
- capacitatea modelului de a detecta cât mai multe daune reale (Recall).

---

## 🔬 Experimente de optimizare (E1–E4)

Au fost realizate patru experimente de optimizare:

- `E1_small_base`
- `E2_lr_up`
- `E3_lr_down`
- `E4_light_aug`

Selecția experimentului „best” s-a făcut pe baza metricii principale: **mAP@50–95(M)** (segmentare / mask), deoarece penalizează puternic localizarea imprecisă și reflectă cel mai bine performanța globală a modelului.

### 📊 Rezultate comparative (best epoch by mAP50–95(M))

| Experiment | Best epoch | mAP@50–95 (M) | mAP@50 (M) | P(M) | R(M) | mAP@50–95 (B) | mAP@50 (B) | P(B) | R(B) |
|------------|-----------:|--------------:|-----------:|-----:|-----:|--------------:|-----------:|-----:|-----:|
| E1_small_base | 6 | 0.5531 | 0.7299 | 0.7788 | 0.7062 | 0.5772 | 0.7422 | 0.7868 | 0.7085 |
| E2_lr_up | 6 | 0.5555 | 0.7363 | 0.7583 | 0.7090 | 0.5790 | 0.7478 | 0.7662 | 0.7131 |
| E3_lr_down | 6 | 0.5514 | 0.7294 | 0.7768 | 0.7066 | 0.5735 | 0.7398 | 0.7827 | 0.7124 |
| **E4_light_aug** | **8** | **0.5680** | **0.7385** | **0.7982** | **0.7098** | **0.5962** | **0.7517** | **0.7998** | **0.7203** |

### Observații analitice

Compararea celor patru configurații evidențiază impactul real al modificărilor de antrenare asupra capacității modelului de a generaliza.

- **E1_small_base** oferă un punct de referință stabil, cu performanțe echilibrate, dar limitate de lipsa unor ajustări suplimentare de fine-tuning.
- **E2_lr_up** crește ușor sensibilitatea modelului (Recall), însă cu o ușoară scădere a preciziei, sugerând un regim de învățare mai agresiv, dar mai puțin stabil.
- **E3_lr_down** stabilizează procesul de învățare, dar nu aduce îmbunătățiri semnificative la localizarea strictă (mAP@50–95), indicând că o rată de învățare prea mică poate limita adaptarea modelului.
- **E4_light_aug** produce cea mai bună performanță globală, crescând simultan localizarea strictă, precizia și recall-ul.

Faptul că E4 îmbunătățește **toate metricile importante în același timp** indică o creștere reală a calității reprezentărilor vizuale învățate de model, nu doar o ajustare superficială a pragurilor de detecție.

---

## 🏆 Selecția modelului final

Pe baza valorii maxime obținute pentru **mAP@50–95(M)**, experimentul **E4_light_aug** a fost ales ca model final al proiectului.

**Motivație tehnică:**

- cel mai bun scor la metrica principală (IoU strict, segmentare);
- performanță superioară și pe Bounding Box;
- echilibru optim între precizie și sensibilitate.

Această alegere reflectă o îmbunătățire reală a robusteții modelului la variații de iluminare, unghiuri și reflexii, fără a introduce supraînvățare.

---

## 📊 Metrici finale (model optimizat)

Metricile finale sunt salvate în:

`results/final_metrics.json`

### 🧩 Segmentare (Mask)
- Precision (P(M)): **0.7982**
- Recall (R(M)): **0.7098**
- mAP@50 (M): **0.7385**
- mAP@50–95 (M): **0.5680**

### 📦 Detecție (Bounding Box)
- Precision (P(B)): **0.7998**
- Recall (R(B)): **0.7203**
- mAP@50 (B): **0.7517**
- mAP@50–95 (B): **0.5962**

---

## 📐 Interpretarea metricilor

Modelul obținut prezintă:

- **Precision ridicată (~0.80)** → număr redus de alarme false  
- **Recall bun (~0.71–0.72)** → majoritatea daunelor sunt detectate  
- **mAP@50 solid (~0.75)** → localizare corectă la nivel practic  
- **mAP@50–95 realist (~0.57–0.60)** → localizarea exactă rămâne dificilă pentru defecte fine

Diferența dintre mAP@50 și mAP@50–95 arată că modelul recunoaște corect zona daunelor, dar conturul precis al zgârieturilor și fisurilor este dificil chiar și pentru anotare umană. Aceasta nu reprezintă o deficiență a modelului, ci o caracteristică a problemei vizuale abordate.

---

## ✅ Concluzie generală

Etapa de optimizare a demonstrat că performanța sistemului poate fi îmbunătățită prin ajustări controlate ale procesului de antrenare, fără a compromite stabilitatea modelului.

Modelul **E4_light_aug** reprezintă cea mai bună variantă obținută în cadrul proiectului, oferind un compromis optim între:

- precizie ridicată,
- sensibilitate bună,
- robustețe la variații vizuale,
- și o localizare realistă a defectelor fine.

Rezultatele obținute sunt coerente, justificabile din punct de vedere tehnic și aliniate cu dificultatea reală a problemei. Sistemul implementat demonstrează o aplicare practică solidă a rețelelor neuronale în domeniul Computer Vision și constituie o bază robustă pentru dezvoltări viitoare, precum extinderea dataset-ului, segmentarea mai precisă sau integrarea în aplicații video.

## 🚀 Posibile direcții de dezvoltare

* extinderea dataset-ului și rebalansarea claselor;
* fine-tuning dedicat pentru defecte fine (`scratch`, `crack`);
* inferență pe secvențe video;
* analiză comparativă cu alte arhitecturi (YOLO variants / Faster R-CNN / RetinaNet).

---

> **Car Damage Detection System** reprezintă o aplicație practică solidă a rețelelor neuronale în domeniul computer vision.




