
---
<div align="center">

# </h>🚗 **CAR DAMAGE DETECTION SYSTEM**</h>

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
* inferență reală într-o aplicație funcțională.

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
├── config/
│   └── car_damage.yaml
│
├── data/
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   │
│   └── images_enhanced/
│       ├── train/
│       ├── val/
│       └── test/
│
├── models/
│   └── trained_model.pt
│
├── results/
│   ├── training_history.csv
│   ├── test_metrics.json
│   └── hyperparameters.yaml
│
├── runs/
│   └── rn_train/
│
├── docs/
│   ├── loss_curve.png
│   └── screenshots/
│       └── inference_real.png
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

Această etapă a avut rolul de a asigura **calitatea datelor**, fără a introduce artificii inutile.

### Aspecte cheie:

* dataset organizat strict în format YOLO;
* separare clară train / validation / test;
* etichete verificate manual;
* fără augmentări agresive.

### Preprocesare utilizată

✔ **ENHANCE (soft)**
✔ evidențiere defecte fine
✔ păstrarea fidelității imaginii originale

Această abordare minimizează riscul de **overfitting artificial**.

---
Perfect, rămânem **exact cu structura ta** 👍
Tot ce trebuie acum este să **integrezi elegant pozele** (grafic + diagramă) astfel încât README-ul să fie:

* 📘 academic (prof-proof)
* 🎬 cinematic (prima pagină „lovește”)
* 🧠 clar (pozele explică textul, nu îl dublează)

Mai jos îți dau **VARIANTA FINALĂ DE INSERARE A POZELOR**, **gata de copiat** în README-ul tău.
Nu schimb nimic din conținutul tău, doar îl **ridic vizual**.

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

# 🖥️ Inferență reală – demonstrație finală

Aplicația finală permite rularea inferenței YOLO pe imagini reale, utilizând același pipeline definit în etapele anterioare.

<p align="center">
  <img src="docs/screenshots/inference_real.png" width="90%">
</p>

<p align="center">
  <i>Figura 3 – Exemplu de inferență reală: imagine originală, preprocesată și rezultatul YOLO</i>
</p>

Funcționalități demonstrate:

* încărcarea unei imagini reale;
* aplicarea automată a preprocesării ENHANCE;
* detectarea daunelor prin YOLO;
* afișarea bounding box-urilor și a scorurilor de încredere.

---

## ⚠️ Limitări și observații

* confuzie între clase vizual similare (`scratch` vs `crack`);
* sensibilitate la variații de iluminare;
* distribuție neechilibrată a claselor.

---

## 🚀 Posibile direcții de dezvoltare

* extinderea dataset-ului;
* fine-tuning dedicat pentru defecte fine;
* inferență pe secvențe video;
* analiză comparativă cu alte arhitecturi.

---

# ✅ Concluzie

Proiectul demonstrează:

* înțelegerea completă a pipeline-ului RN;
* implementarea unui sistem real, funcțional;
* documentație coerentă, clară și academică.

> **Car Damage Detection System** reprezintă o aplicație practică solidă a rețelelor neuronale în domeniul computer vision.

---


Spune-mi sincer:
👉 *mai sus* sau *asta e bomba finală*?
