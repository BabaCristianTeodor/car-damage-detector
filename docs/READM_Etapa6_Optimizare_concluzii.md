---

# 📘 Etapa 6 – Optimizare, evaluare și concluzii finale

**Disciplina:** Rețele Neuronale
**Instituție:** POLITEHNICA București – FIIR
**Student:** Baba Cristian Teodor
**Proiect:** Detecția daunelor auto utilizând rețele neuronale
**An universitar:** 2025–2026

---

## 1. Scopul Etapei 6

Etapa 6 are ca obiectiv optimizarea modelului de detecție a daunelor auto realizat în Etapa 5, analiza comparativă a mai multor configurații de antrenare și integrarea modelului optimizat în aplicația finală.

Această etapă reprezintă finalizarea ciclului de dezvoltare al proiectului, punând accent pe maturizarea modelului, evaluarea realistă a performanței și formularea concluziilor finale.

---

## 2. Strategia de optimizare

Optimizarea a fost realizată prin rularea a patru experimente distincte, variind parametrii de antrenare ai modelului YOLO.

Criteriul principal de selecție a modelului final a fost:
- **mAP@50–95**, metrică standard pentru detecția de obiecte, care evaluează atât corectitudinea clasificării, cât și precizia localizării pe mai multe praguri IoU.

Metricile Precision și Recall au fost utilizate complementar pentru a analiza comportamentul modelului în raport cu diferite tipuri de defecte.

---

## 3. Experimente de optimizare

Rezultatele experimentelor sunt centralizate în fișierul:

results/optimization_experiments.csv

yaml
Copy code

### Rezultate comparative (ultimul epoch)

| Experiment | mAP@50 | mAP@50–95 | Precision | Recall |
|-----------|--------|-----------|-----------|--------|
| exp1 | 0.49669 | 0.37809 | 0.58170 | 0.46708 |
| exp2 | 0.49669 | 0.37809 | 0.58170 | 0.46708 |
| exp3 | **0.51103** | **0.38338** | 0.56153 | **0.49470** |
| exp4 | 0.38352 | 0.27721 | 0.45227 | 0.43409 |

---

## 4. Selecția modelului optimizat

Pe baza valorii maxime obținute pentru **mAP@50–95**, experimentul **exp3** a fost ales ca model optimizat final.

Alegerea acestui experiment este justificată prin:
- cea mai bună performanță globală de detecție;
- îmbunătățirea mAP@50–95 față de celelalte experimente;
- un recall mai ridicat, important pentru reducerea ratărilor de defecte în scenarii reale.

---

## 5. Modelul final

Modelul optimizat este salvat în:

models/optimized_model.pt

yaml
Copy code

Acest model înlocuiește complet modelul utilizat în Etapa 5 și este utilizat pentru toate etapele de evaluare și inferență din aplicația finală.

---

## 6. Evaluare finală și Confusion Matrix

Evaluarea modelului optimizat a fost realizată pe setul de validare.

Confusion Matrix rezultată este disponibilă în:

docs/confusion_matrix_optimized.png

yaml
Copy code

Analiza Confusion Matrix indică:
- confuzii frecvente între clasele **scratch** și **crack**, cauzate de similitudini vizuale și diferențe subtile de textură;
- performanță foarte bună pentru clasele cu caracteristici vizuale clare (ex. *tire_flat*, *glass_shatter*);
- impactul negativ al dezechilibrului dataset-ului asupra claselor cu puține exemple.

---

## 7. Integrarea în aplicația software

Aplicația UI a fost actualizată pentru a utiliza exclusiv modelul optimizat:

models/optimized_model.pt

css
Copy code

Funcționalitatea a fost verificată prin inferență reală, iar un screenshot demonstrativ este salvat în:

docs/screenshots/inference_optimized.png

yaml
Copy code

---

## 8. Metrici finale

Metricile finale ale modelului optimizat sunt salvate în:

results/final_metrics.json

markdown
Copy code

### Valori obținute (macro):

- Precision: **0.568**
- Recall: **0.495**
- mAP@50: **0.512**
- mAP@50–95: **0.383**

---

## 9. Concluzii finale și interpretarea scorurilor obținute

Scorurile obținute reflectă un compromis realist între performanță și complexitatea problemei abordate.

Valoarea **mAP@50–95 = 0.383** indică o capacitate bună de detecție într-un context dificil, caracterizat de:
- clase cu defecte vizual similare;
- variații mari de iluminare și unghi;
- distribuție dezechilibrată a datelor între clase.

Diferența dintre **mAP@50 (0.512)** și **mAP@50–95 (0.383)** sugerează că modelul detectează corect majoritatea obiectelor la praguri IoU mai relaxate, însă întâmpină dificultăți în localizarea extrem de precisă a defectelor mici sau alungite (ex. *scratch*).

Valoarea **Precision = 0.568** arată că majoritatea predicțiilor pozitive sunt corecte, în timp ce **Recall = 0.495** evidențiază faptul că o parte din defecte nu sunt detectate, în special în cazul claselor minoritare. Acest comportament este explicabil prin:
- numărul redus de exemple pentru anumite clase;
- suprapunerea vizuală între tipuri diferite de daune.

În ansamblu, modelul optimizat reprezintă o îmbunătățire clară față de versiunea anterioară și oferă un echilibru bun între precizie și capacitatea de generalizare. Rezultatele sunt conforme cu așteptările pentru un dataset real, neideal, și demonstrează corect aplicarea tehnicilor de optimizare și evaluare în cadrul unui proiect de rețele neuronale.

Proiectul oferă o implementare completă și funcțională a unui sistem de detecție a daunelor auto, fiind extensibil prin adăugarea de date suplimentare, rebalansarea claselor și rafinarea strategiilor de antrenare.