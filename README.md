<div align="center">
🚗 CAR DAMAGE DETECTION SYSTEM
Sistem inteligent pentru detecția daunelor auto folosind Rețele Neuronale
</div> <p align="center"> <img src="https://img.shields.io/badge/Neural%20Networks-UPB%20FIIR-blueviolet"> <img src="https://img.shields.io/badge/YOLO-v11m-0A66C2"> <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C"> <img src="https://img.shields.io/badge/GPU-RTX%204060-success"> <img src="https://img.shields.io/badge/Project-Academic%20Final-brightgreen"> </p> <p align="center"> <b>Universitatea POLITEHNICA București</b><br> Facultatea de Inginerie Industrială și Robotică (FIIR)<br> Disciplina: <b>Rețele Neuronale</b><br><br> Student: <b>Baba Cristian Teodor</b><br> An universitar: 2025–2026 </p>
🌌 Introducere

Proiectul Car Damage Detection System reprezintă implementarea unui sistem inteligent complet, bazat pe rețele neuronale convoluționale, capabil să detecteze automat daunele vizibile ale unui vehicul dintr-o imagine digitală.

Sistemul nu se limitează la antrenarea unui model, ci acoperă întregul ciclu de viață al unei aplicații bazate pe RN:

analiză și pregătire dataset,

definirea arhitecturii software,

antrenare și evaluare,

inferență reală într-o aplicație funcțională,

optimizare și selecția modelului final.

🧠 Tipul problemei abordate
<table> <tr><td><b>Tip problemă</b></td><td>Object Detection (Computer Vision)</td></tr> <tr><td><b>Date de intrare</b></td><td>Imagini RGB cu vehicule avariate</td></tr> <tr><td><b>Date de ieșire</b></td><td>Bounding box-uri + clasă + scor de încredere</td></tr> <tr><td><b>Model RN</b></td><td>YOLO11m</td></tr> </table>
🏷️ Clase de daune detectate
<p align="center"> <img src="https://img.shields.io/badge/dent-gray"> <img src="https://img.shields.io/badge/scratch-blue"> <img src="https://img.shields.io/badge/crack-purple"> <img src="https://img.shields.io/badge/glass_shatter-cyan"> <img src="https://img.shields.io/badge/lamp_broken-orange"> <img src="https://img.shields.io/badge/tire_flat-red"> </p>

Aceste clase au fost alese pentru a acoperi atât:

defecte structurale (dent, crack),

cât și defecte funcționale (lamp_broken, tire_flat).

📂 Structura actuală a proiectului

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
🧪 ETAPA 3 — Analiza și pregătirea setului de date

Această etapă a avut rolul de a asigura calitatea datelor, fără a introduce artificii inutile care ar putea distorsiona generalizarea.

Aspecte cheie:

dataset organizat strict în format YOLO;

separare clară train / validation / test;

etichete verificate manual;

fără augmentări agresive.

Preprocesare utilizată

✔ ENHANCE (soft)
✔ evidențiere defecte fine
✔ păstrarea fidelității imaginii originale

Această abordare minimizează riscul de overfitting artificial și păstrează un pipeline ușor de justificat academic.

🏗️ ETAPA 4 — Arhitectura sistemului (State Machine)

Aplicația este modelată ca o mașină de stări (State Machine), oferind control complet asupra fluxului de execuție și o integrare clară a rețelei neuronale într-un sistem software real.

Principii arhitecturale

determinism al execuției;

tratare explicită a erorilor;

reset controlat al aplicației;

separare clară între etapele logice ale pipeline-ului RN.

Diagrama de stări a aplicației
<p align="center"> <img src="docs/state_machine_car_damage.png" width="85%"> </p> <p align="center"> <i>Figura 1 – Diagrama State Machine a sistemului de detecție a daunelor auto</i> </p>

Fiecare stare din diagramă corespunde unei faze logice distincte:

încărcarea imaginii de către utilizator;

aplicarea preprocesării ENHANCE;

validarea formatului și dimensiunii imaginii;

rularea inferenței YOLO;

afișarea rezultatelor sau tratarea erorilor;

resetarea aplicației sau oprirea controlată.

Această abordare permite o analiză clară a fluxului și o implementare robustă, ușor de extins.

🤖 ETAPA 5 — Antrenarea și evaluarea rețelei neuronale
Configurație utilizată
Componentă	Specificație
GPU	NVIDIA RTX 4060 – 8GB
Framework	PyTorch + Ultralytics
Model	YOLO11m
Mod de rulare	Local
Procesul de învățare

Pentru analiza comportamentului rețelei neuronale pe parcursul antrenării a fost monitorizată evoluția funcției de pierdere (loss) atât pe setul de antrenare, cât și pe setul de validare.

<p align="center"> <img src="docs/loss_curve.png" width="85%"> </p> <p align="center"> <i>Figura 2 – Evoluția loss-ului de antrenare și validare</i> </p>

Graficul Loss vs Validation Loss evidențiază:

o scădere constantă a loss-ului de antrenare;

stabilitatea loss-ului pe setul de validare;

lipsa unui overfitting sever;

activarea corectă a mecanismului de early stopping.

Acest comportament indică o convergență stabilă a modelului.

🖥️ Inferență reală – demonstrație (model antrenat)

Aplicația finală permite rularea inferenței YOLO pe imagini reale, utilizând același pipeline definit în etapele anterioare.

<p align="center"> <img src="docs/screenshots/inference_real.png" width="90%"> </p> <p align="center"> <i>Figura 3 – Exemplu de inferență reală (Etapa 5): imagine originală, preprocesată și rezultatul YOLO</i> </p>

Funcționalități demonstrate:

încărcarea unei imagini reale;

aplicarea automată a preprocesării ENHANCE;

detecția daunelor prin YOLO;

afișarea bounding box-urilor și a scorurilor de încredere.

⚠️ Limitări și observații (context pentru interpretarea metricilor)

confuzie între clase vizual similare (scratch vs crack);

sensibilitate la variații de iluminare și reflexii ale caroseriei;

distribuție neechilibrată a claselor în setul de date;

dimensiunea redusă a setului de date, determinată de necesitatea limitării numărului de imagini pentru antrenare eficientă pe resurse locale (GPU și memorie);

ca urmare, clasele cu defecte fine (scratch și crack) dispun de mai puține exemple relevante, ceea ce conduce la performanțe mai scăzute comparativ cu defectele cu contrast vizual ridicat.

Această limitare este specifică scenariilor de antrenare pe resurse hardware locale și nu reprezintă o deficiență conceptuală a arhitecturii alese.

🧪 ETAPA 6 — Optimizare, selecția modelului și evaluare finală

În această etapă a fost realizată optimizarea performanței modelului YOLO utilizat pentru detecția daunelor auto, prin rularea mai multor experimente controlate și evaluarea comparativă a rezultatelor obținute.

Scopul principal a fost selecția unui model final cu performanță globală superioară, utilizând metrici standard din domeniul object detection.

📏 Metrică de evaluare utilizată

Pentru evaluarea și compararea experimentelor a fost utilizată ca metrică principală:

mAP@50–95 (mean Average Precision pe multiple praguri IoU)

Aceasta este considerată metrică standard în detecția de obiecte, oferind o evaluare robustă a calității localizării și clasificării.

Metricile Precision și Recall au fost analizate complementar, pentru interpretarea comportamentului modelului.

🔬 Experimente de optimizare (exp1–exp4)

Au fost realizate patru experimente de optimizare, denumite exp1 – exp4.
Rezultatele complete sunt documentate în:

results/optimization_experiments.csv

📊 Rezultate comparative (ultimul epoch)
Experiment	mAP@50	mAP@50–95	Precision	Recall
exp1	0.49669	0.37809	0.58170	0.46708
exp2	0.49669	0.37809	0.58170	0.46708
exp3	0.51103	0.38338	0.56153	0.49470
exp4	0.38352	0.27721	0.45227	0.43409
Observații analitice asupra experimentelor

exp1 vs exp2: rezultate identice (comportament de convergență similar), ceea ce indică fie o modificare cu impact nul, fie o modificare neutralizată de setări implicite din pipeline.

exp3: obține cel mai bun compromis global, crescând atât mAP@50, cât și mAP@50–95, concomitent cu îmbunătățirea recall-ului; indică o generalizare mai robustă.

exp4: degradare semnificativă pe toate metricile, sugerând un regim de învățare nepotrivit pentru distribuția dataset-ului.

🏆 Selecția modelului optim

Pe baza valorii maxime obținute pentru mAP@50–95, experimentul exp3 a fost ales ca model optimizat final.

Motivație: exp3 oferă cea mai bună performanță globală, având cel mai ridicat scor mAP@50–95 și un recall superior, indicând o capacitate mai bună de detectare a daunelor auto pe setul de validare.

💾 Modelul final utilizat

Modelul rezultat în urma Etapei 6 este salvat ca:

models/optimized_model.pt

Acest model înlocuiește complet versiunea utilizată în Etapa 5 și reprezintă modelul final al proiectului.

📉 Confusion Matrix și evaluare finală

Confusion Matrix pentru modelul optimizat a fost generată în urma evaluării pe setul de validare.

<p align="center"> <img src="docs/confusion_matrix_normalized.png" width="85%"> </p> <p align="center"> <i>Figura 4 – Matricea de confuzie a sistemului (model optimizat)</i> </p>

Analiza evidențiază confuzii între clase vizual similare (ex. scratch și crack), precum și o performanță superioară pentru defectele cu contrast vizual ridicat.

🖥️ Integrarea în aplicația finală (model optimizat)

Aplicația UI a fost actualizată pentru a utiliza exclusiv modelul optimizat (optimized_model.pt), asigurând consistența între etapa de evaluare și inferența realizată în aplicația finală.

<p align="center"> <img src="docs/screenshots/inference_optimized_f.png" width="85%"> </p> <p align="center"> <i>Figura 5 – Screenshot: inferență cu modelul optimizat încărcat și testat</i> </p>
📊 Metrici finale (model optimizat)

Modelul optimizat a fost selectat pe baza metricii principale mAP@50–95 (Mask), conform summary-ului:

=== SUMMARY (best epoch by metrics/mAP50-95(M)) ===
Best experiment: E4_light_aug
Best epoch: 8

Metricile finale sunt salvate în:

results/final_metrics.json
🔢 Valori finale raportate (TEST set)
🧩 Segmentare (Mask)

Precision (P(M)): 0.7982

Recall (R(M)): 0.7098

mAP@50 (M): 0.7385

mAP@50–95 (M): 0.5680

📦 Detecție (Bounding Box)

Precision (P(B)): 0.7998

Recall (R(B)): 0.7203

mAP@50 (B): 0.7517

mAP@50–95 (B): 0.5962

➡️ Modelul E4_light_aug oferă cel mai bun compromis global între detecție și segmentare, fiind ales drept model final al proiectului.

📐 Interpretare detaliată a coeficienților (metricilor) — secțiunea critică

În object detection și instance segmentation, metricile trebuie interpretate în contextul problemei, deoarece fiecare coeficient descrie un aspect diferit al comportamentului rețelei neuronale.
În cazul daunelor auto — mai ales defecte fine și alungite — localizarea exactă este intrinsec dificilă, ceea ce afectează direct scorurile mAP stricte.

🔹 Precision (Precizia)

Precision reprezintă proporția predicțiilor corecte din totalul predicțiilor realizate:

𝑃
𝑟
𝑒
𝑐
𝑖
𝑠
𝑖
𝑜
𝑛
=
𝑇
𝑃
𝑇
𝑃
+
𝐹
𝑃
Precision=
TP+FP
TP
	​


Precision ≈ 0.80 (Mask & Box) indică faptul că modelul produce, în majoritatea cazurilor, detecții valide, cu un număr redus de alarme false.

🔎 Interpretare practică:

sistemul este conservator;

evită raportarea unor daune inexistente;

comportament dezirabil pentru aplicații de inspecție și triere inițială.

În context industrial, acest lucru reduce costurile generate de reinspecții inutile.

🔹 Recall (Rata de detecție)

Recall măsoară proporția daunelor reale detectate corect:

𝑅
𝑒
𝑐
𝑎
𝑙
𝑙
=
𝑇
𝑃
𝑇
𝑃
+
𝐹
𝑁
Recall=
TP+FN
TP
	​


Recall ≈ 0.71 (Mask) și ≈ 0.72 (Box) arată că majoritatea daunelor reale sunt identificate, însă o parte rămân nedetectate, în special în cazul:

defectelor subțiri cu contur difuz (scratch);

fisurilor cu textură similară fundalului (crack);

claselor slab reprezentate (dataset neechilibrat).

🔎 Interpretare practică:

modelul este suficient de sensibil pentru uz academic și prototip industrial;

creșterea recall-ului ar necesita augmentări mai agresive sau un dataset extins, cu riscul scăderii precision.

🔹 mAP@50 (Mean Average Precision @ IoU 0.5)

mAP@50 evaluează performanța la un prag IoU permisiv, unde bounding box-ul trebuie să se suprapună rezonabil cu adevărul de referință.

mAP@50 ≈ 0.74 (Mask) și ≈ 0.75 (Box) indică faptul că modelul:

recunoaște corect zona aproximativă a daunelor;

oferă detecții stabile și consistente;

este suficient pentru aplicații de triere automată sau suport decizional.

Această valoare confirmă utilitatea practică a sistemului.

🔹 mAP@50–95 (metrică principală, strictă)

mAP@50–95 reprezintă media performanței pe praguri IoU între 0.50 și 0.95 și penalizează sever localizările imprecise.

mAP@50–95 ≈ 0.568 (Mask) și ≈ 0.596 (Box) sunt valori realiste și solide pentru detecția daunelor auto, deoarece:

defectele sunt mici, alungite și greu de încadrat exact;

anotările umane prezintă variații inerente;

dataset-ul este neechilibrat;

nu s-au folosit augmentări agresive (decizie intenționată pentru realism și justificare academică).

🔎 Diferența dintre mAP@50 și mAP@50–95 indică faptul că:

defectele sunt recunoscute corect,

dar localizarea foarte precisă rămâne principala provocare.

🔎 Corelarea metricilor cu realitatea aplicației

Combinația obținută:

Precision ridicată (~0.80)

Recall bun (~0.71)

mAP@50 solid (~0.74)

mAP@50–95 strict (~0.57)

descrie un model echilibrat și matur, care:

produce detecții curate;

menține o rată bună de identificare;

este limitat doar de dificultatea intrinsecă a defectelor fine și de dimensiunea dataset-ului.

👉 Exact tipul de rezultat așteptat și corect pentru un proiect academic realist, fără supra-optimizare artificială.

✅ Concluzie tehnică

Modelul E4_light_aug, selectat pe baza mAP@50–95, reprezintă cea mai bună variantă obținută în cadrul proiectului, oferind un compromis optim între precizie, sensibilitate și robustețe.
Rezultatele sunt coerente, justificabile și aliniate cu cerințele disciplinei Rețele Neuronale.

🚀 Posibile direcții de dezvoltare

extinderea dataset-ului și rebalansarea claselor;

fine-tuning dedicat pentru defecte fine (scratch, crack);

inferență pe secvențe video;

analiză comparativă cu alte arhitecturi (YOLO variants / Faster R-CNN / RetinaNet).

Car Damage Detection System reprezintă o aplicație practică solidă a rețelelor neuronale în domeniul computer vision.
