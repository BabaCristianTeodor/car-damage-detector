
---

# 📘 ETAPA 5 – Antrenarea și Evaluarea Modelului de Rețea Neuronală

**Disciplina:** Rețele Neuronale
**Facultatea:** Inginerie Industrială și Robotică – POLITEHNICA București
**Proiect:** Sistem inteligent de detecție a daunelor auto
**Student:** Baba Cristian Teodor
**Model RN:** YOLO11m (detecție obiecte)
**Cadru de lucru:** PyTorch + Ultralytics YOLO
**Hardware:** NVIDIA RTX 4060 Laptop GPU (8 GB VRAM)

---

## 1. Scopul etapei

Scopul **Etapei 5** este **antrenarea efectivă a modelului de rețea neuronală**, evaluarea performanței acestuia pe un set de test independent și **integrarea modelului antrenat într-o aplicație cu inferență reală**, conform arhitecturii definite în Etapa 4.

Această etapă demonstrează trecerea de la:

* arhitectură conceptuală
* model neantrenat / dummy

la un **sistem funcțional complet**, capabil să ruleze inferență reală pe imagini reale.

---

## 2. Continuitate față de Etapa 4 (obligatoriu)

Antrenarea respectă **State Machine-ul** definit anterior, având următoarea corespondență:

| Stare (Etapa 4) | Implementare în Etapa 5                   |
| --------------- | ----------------------------------------- |
| START_SYSTEM    | Inițializare aplicație și încărcare model |
| WAIT_IMAGE      | Așteptare încărcare imagine din UI        |
| ENHANCE         | Aplicare pre-procesare (enhance soft)     |
| VALIDATE_IMAGE  | Verificare format, rezoluție              |
| RN_INFERENCE    | Inferență YOLO cu model antrenat          |
| DRAW_RESULTS    | Desenare bounding box-uri                 |
| LOG / ERROR     | Salvare log și tratare erori              |
| END             | Eliberare resurse                         |

Diagrama utilizată este cea prezentată în `docs/state_machine.png`.

---

## 3. Dataset și clase utilizate

### 3.1 Clasele de detecție

Modelul este antrenat pentru **6 clase distincte de daune auto**:

1. `dent` – îndoitură
2. `scratch` – zgârietură
3. `crack` – fisură
4. `glass_shatter` – sticlă spartă
5. `lamp_broken` – far spart
6. `tire_flat` – pană

### 3.2 Organizarea datelor

Structura dataset-ului respectă formatul YOLO:

```
data/
└── images/
    ├── train/
    ├── val/
    └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

Proporțiile utilizate:

* **70% train**
* **15% validation**
* **15% test**

---

## 4. Configurarea și antrenarea modelului

### 4.1 Model utilizat

* Arhitectură: **YOLO11m**
* Număr parametri: ~20 milioane
* Tip sarcină: detecție obiecte (bounding boxes)

Modelul a fost antrenat **de la zero**, fără fine-tuning pe un model deja antrenat pentru aceeași sarcină.

---

### 4.2 Hiperparametri utilizați

| Hiperparametru | Valoare                 | Justificare                     |
| -------------- | ----------------------- | ------------------------------- |
| Epochs         | 180 (cu early stopping) | Permite convergență completă    |
| Batch size     | Adaptiv (GPU 8GB)       | Echilibru memorie / stabilitate |
| Optimizer      | Adam                    | Convergență stabilă             |
| Learning rate  | implicit YOLO           | Optimizat pentru detecție       |
| Early stopping | patience = 25           | Evită overfitting               |
| Augmentare     | implicit YOLO           | Generalizare mai bună           |

---

## 5. Evoluția procesului de antrenare

### 5.1 Grafic loss vs val_loss

Graficul de mai jos (`docs/loss_curve.png`) evidențiază:

* scădere constantă a **train loss**
* stabilizarea **val loss**
* oprire automată prin **early stopping**

➡️ Acest comportament indică **convergență corectă**, fără overfitting sever.

---

## 6. Evaluarea modelului pe setul de test

Evaluarea a fost realizată pe setul **test**, complet separat de antrenare.

### 6.1 Metrici globale

* **Precision (mean):** ~0.57
* **Recall (mean):** ~0.52
* **F1-score (mean):** ~0.54
* **mAP@50:** ~0.49
* **mAP@50–95:** ~0.33

Aceste valori sunt **realiste pentru un dataset complex, neechilibrat**, și confirmă funcționarea corectă a modelului.

Metricile sunt salvate în:

```
results/test_metrics.json
```

---

## 7. Integrarea modelului în aplicația finală

### 7.1 Interfață cu inferență reală

Modelul antrenat (`models/trained_model.pt`) este integrat într-o aplicație web academică, care permite:

* încărcarea unei imagini reale
* aplicarea unui **enhance soft**
* rularea inferenței YOLO
* afișarea bounding box-urilor și scorurilor

Un exemplu de inferență reală este salvat în:

```
docs/screenshots/inference_real.png
```

### 7.2 Pre-procesare (Enhance)

Pentru a evita degradarea performanței, s-a utilizat un **enhance moderat**, care:

* îmbunătățește contrastul local
* nu distorsionează texturile fine
* păstrează informația utilă pentru YOLO

---

## 8. Analiza erorilor (context aplicație)

### Observații:

* Clasele `crack` și `scratch` sunt uneori confundate
* Cauză: texturi similare și limite neclare între defecte
* Clasele rare (`glass_shatter`, `tire_flat`) au scoruri mai bune datorită contrastului vizual puternic

### Impact:

* **False positives** → acceptabile (inspecție manuală)
* **False negatives** → mai critice

### Măsuri propuse:

1. Creșterea numărului de exemple pentru clasele fine
2. Ajustarea pragului de încredere
3. Augmentare direcționată pe zgârieturi subtile

---

## 9. Structura finală a proiectului (Etapa 5)

```
models/
├── trained_model.pt

results/
├── training_history.csv
├── test_metrics.json
├── hyperparameters.yaml

docs/
├── loss_curve.png
├── state_machine.png
└── screenshots/
    └── inference_real.png
```

---

## 10. Concluzie

Etapa 5 confirmă că:

* modelul RN a fost **antrenat corect**
* evaluarea este **realistă și documentată**
* inferența este **funcțională și demonstrată**
* arhitectura din Etapa 4 este respectată integral

Sistemul rezultat este un **Sistem Inteligent complet**, pregătit pentru utilizare reală și extensii viitoare.

---

