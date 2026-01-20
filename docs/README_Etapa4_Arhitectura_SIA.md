# 📘 Etapa 4 – Arhitectura Sistemului bazat pe Rețele Neuronale

**Disciplina:** Rețele Neuronale  
**Instituție:** Universitatea POLITEHNICA București – FIIR  
**Student:** Baba Cristian Teodor  
**Proiect:** Detecția automată a daunelor auto din imagini  
**An universitar:** 2025–2026  

---

## 1. Scopul Etapei 4

Scopul Etapei 4 este **definirea arhitecturii funcționale a sistemului software** bazat pe rețele neuronale, precum și descrierea **fluxului de execuție intern**, utilizând un model de tip **State Machine**.

Această etapă NU urmărește:

- optimizarea performanței;
- obținerea unor metrici ridicate;
- evaluarea finală a modelului.

Obiectivul este **demonstrarea înțelegerii arhitecturii RN și a integrării acesteia într-un sistem funcțional**.

---

## 2. Nevoie reală și soluția propusă prin SIA

| Nevoie reală concretă | Soluția oferită de sistem | Modul software implicat |
|-----------------------|---------------------------|-------------------------|
| Detectarea automată a daunelor auto din imagini | Analiza imaginilor utilizând o rețea neuronală YOLO | Modul RN |
| Reducerea timpului de analiză manuală | Inferență automată în timp real | Modul RN |
| Interacțiune facilă cu utilizatorul | Interfață grafică pentru încărcare și afișare rezultate | Modul UI |
| Control sigur al fluxului aplicației | Arhitectură bazată pe State Machine | Modul logic |

---

## 3. Tipul arhitecturii utilizate

Sistemul este implementat sub forma unei **mașini de stări (State Machine)**, deoarece:

- aplicația este declanșată de evenimente (încărcarea unei imagini);
- fluxul este secvențial și determinist;
- pot fi tratate explicit cazurile de eroare;
- aplicația poate reveni controlat într-o stare inițială.

Această abordare este potrivită pentru aplicații de inferență offline / semi-interactive.

---

## 4. Descriere generală a fluxului de execuție

Fluxul complet al aplicației este următorul:

Start sistem
→ Așteaptă încărcarea unei imagini
→ Preprocesare (ENHANCE)
→ Validare imagine
→ Inferență YOLO
→ Afișare rezultate / Tratare erori
→ Reset / Oprire aplicație


Fiecare etapă corespunde unei **stări distincte** din diagrama State Machine.

Diagrama completă a State Machine-ului este disponibilă în:
docs/state_machine.png


---

## 5. Descrierea stărilor din State Machine

### 5.1 Starea START

**Rol:**  
Inițializează sistemul și resursele necesare (model RN, CPU/GPU, configurări).

**Tranziție:**  
→ `Așteaptă încărcarea unei imagini`

---

### 5.2 Starea „Așteaptă încărcarea unei imagini”

**Rol:**  
Sistemul se află în stare pasivă, așteptând acțiunea utilizatorului.

**Evenimente posibile:**

- utilizatorul încarcă o imagine → trecere la preprocesare;
- utilizatorul oprește aplicația → închidere controlată.

---

### 5.3 Starea „Aplică filtre ENHANCE”

**Rol:**  
Aplică singura etapă de preprocesare utilizată în proiect: **ENHANCE**.

Operația ENHANCE:
- îmbunătățește contrastul local;
- evidențiază defectele fine;
- nu modifică geometria imaginii.

**Tranziție:**  
→ `Verifică format, rezoluție, dimensiune`

---

### 5.4 Starea „Verifică format, rezoluție, dimensiune”

**Rol:**  
Asigură validitatea imaginii de intrare.

**Verificări efectuate:**

- format imagine valid (JPG / PNG);
- imagine necoruptă;
- dimensiuni compatibile cu inferența YOLO.

**Tranziții:**

- imagine validă → `Rulează inferența YOLO`;
- fișier invalid → `Afișează eroare și salvează log`.

---

### 5.5 Starea „Rulează inferența YOLO”

**Rol:**  
Execută inferența utilizând rețeaua neuronală YOLO.

**Output generat:**

- bounding box-uri;
- scoruri de încredere;
- clasele detectate.

**Tranziții:**

- inferență reușită → `Desenează bounding box-uri și salvează rezultatul`;
- eroare RN / hardware → `Afișează eroare și salvează log`.

---

### 5.6 Starea „Desenează bounding box-uri și salvează rezultatul”

**Rol:**

- suprapune bounding box-urile pe imagine;
- salvează rezultatul final;
- afișează rezultatul utilizatorului.

**Tranziție:**  
→ revenire la `Așteaptă încărcarea unei imagini`  
sau → `Oprire aplicație`.

---

### 5.7 Starea „Afișează eroare și salvează log”

**Rol:**  
Gestionează situațiile de eroare:

- fișiere invalide;
- erori de inferență;
- probleme hardware.

**Acțiuni:**

- afișare mesaj de eroare;
- salvare informații în fișiere de log.

**Tranziție:**  
→ reset către `Așteaptă încărcarea unei imagini`  
sau → `Oprire aplicație`.

---

### 5.8 Starea „Oprire aplicație / eliberare resurse”

**Rol:**

- eliberarea memoriei;
- închiderea sesiunilor CPU/GPU;
- oprirea controlată a aplicației.

**Tranziție:**  
→ `End`

---

## 6. Modulele sistemului

Sistemul este organizat în trei module principale:

1. **Modul Data / Preprocesare** – gestionarea imaginilor și aplicarea filtrului ENHANCE  
2. **Modul Rețea Neuronală (RN)** – inferența YOLO pentru detectarea daunelor  
3. **Modul UI** – interfața cu utilizatorul și afișarea rezultatelor  

---

## 7. Contribuția originală a studentului

Contribuția originală în cadrul proiectului este de **peste 40%** și constă în:

- definirea completă a arhitecturii software;
- proiectarea State Machine-ului;
- integrarea logică a rețelei neuronale într-un sistem funcțional;
- definirea fluxurilor de eroare și reset.

Tipul contribuției:
- [x] Proiectare arhitecturală
- [x] Implementare logică
- [x] Integrare RN + UI

---

## 8. Structura repository-ului (relevantă pentru Etapa 4)

├── data/
├── src/
│ ├── neural_network/
│ └── app/
├── docs/
│ └── state_machine.png
├── models/
└── README_Etapa4_Arhitectura_SIA.md

---

## 9. Concluzie

Etapa 4 definește arhitectura logică și funcțională a sistemului de detecție a daunelor auto, demonstrând:

- înțelegerea fluxului intern al aplicației;
- utilizarea corectă a unei arhitecturi bazate pe State Machine;
- integrarea unei rețele neuronale YOLO într-un sistem software coerent.

Această etapă constituie fundamentul necesar pentru **Etapa 5 – Antrenarea și evaluarea modelului**.