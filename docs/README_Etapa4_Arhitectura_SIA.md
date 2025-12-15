
---

# 📘 Etapa 4 – Arhitectura Sistemului bazat pe Rețele Neuronale

**Disciplina:** Rețele Neuronale
**Instituție:** Universitatea POLITEHNICA București – FIIR
**Student:** Baba Cristian Teodor
**Proiect:** Detecția automată a daunelor auto din imagini
**An universitar:** 2026–2026

---

## 1. Scopul Etapei 4

Scopul Etapei 4 este **definirea arhitecturii funcționale a sistemului software** bazat pe rețele neuronale, precum și descrierea **fluxului de execuție intern**, utilizând un model de tip **State Machine**.

Această etapă NU urmărește:

* optimizarea performanței,
* obținerea unor metrici ridicate,
* evaluarea finală a modelului.

Obiectivul este **demonstrarea înțelegerii arhitecturii RN și a integrării acesteia într-un sistem funcțional**.

---

## 2. Tipul arhitecturii utilizate

Sistemul este implementat sub forma unei **mașini de stări (State Machine)**, deoarece:

* aplicația este declanșată de evenimente (încărcarea unei imagini);
* fluxul este secvențial și determinist;
* pot fi tratate explicit cazurile de eroare;
* aplicația poate reveni controlat într-o stare inițială.

Această abordare este potrivită pentru aplicații de inferență offline / semi-interactive.

---

## 3. Descriere generală a fluxului de execuție

Fluxul complet al aplicației este următorul:

```
Start sistem
 → Așteaptă încărcarea unei imagini
 → Preprocesare (ENHANCE)
 → Validare imagine
 → Inferență YOLO
 → Afișare rezultate / Tratare erori
 → Reset / Oprire aplicație
```

Fiecare etapă corespunde unei **stări distincte** din diagrama State Machine.

---

## 4. Descrierea stărilor din State Machine

### 4.1 Starea START

**Rol:**
Inițializează sistemul și resursele necesare (model RN, GPU/CPU, configurări).

**Tranziție:**
→ `Așteaptă încărcarea unei imagini`

---

### 4.2 Starea „Așteaptă încărcarea unei imagini”

**Rol:**
Sistemul se află în stare pasivă, așteptând acțiunea utilizatorului.

**Evenimente posibile:**

* utilizatorul încarcă o imagine → trecere la preprocesare;
* utilizatorul oprește aplicația → închidere controlată.

---

### 4.3 Starea „Aplică filtre ENHANCE”

**Rol:**
Aplică singura etapă de preprocesare utilizată în proiect: **ENHANCE**.

Operația ENHANCE:

* îmbunătățește contrastul local;
* evidențiază defectele fine;
* nu modifică geometria imaginii.

**Tranziție:**
→ `Verifică format, rezoluție, dimensiune`

---

### 4.4 Starea „Verifică format, rezoluție, dimensiune”

**Rol:**
Asigură validitatea imaginii de intrare.

**Verificări efectuate:**

* format imagine valid (ex. JPG/PNG);
* imagine necoruptă;
* dimensiuni acceptabile pentru inferență.

**Tranziții:**

* imagine validă → `Rulează inferența YOLO`;
* fișier invalid / corupt → `Afișează eroare și salvează log`.

---

### 4.5 Starea „Rulează inferența YOLO”

**Rol:**
Execută inferența utilizând rețeaua neuronală YOLO.

**Output generat:**

* bounding box-uri;
* scoruri de încredere;
* clasele detectate.

**Tranziții:**

* inferență reușită → `Desenează bounding box-uri și salvează rezultatul`;
* eroare RN / GPU / timeout → `Afișează eroare și salvează log`.

---

### 4.6 Starea „Desenează bounding box-uri și salvează rezultatul”

**Rol:**

* suprapune bounding box-urile pe imagine;
* salvează rezultatul final;
* afișează rezultatul utilizatorului.

**Tranziție:**
→ revenire la `Așteaptă încărcarea unei imagini` (pentru o nouă inferență)
sau → `Oprire aplicație`.

---

### 4.7 Starea „Afișează eroare și salvează log”

**Rol:**
Gestionează toate situațiile de eroare:

* fișiere invalide;
* erori de inferență;
* probleme hardware (GPU / timeout).

**Acțiuni:**

* afișare mesaj de eroare;
* salvare informații în log.

**Tranziție:**
→ reset către `Așteaptă încărcarea unei imagini`
sau → `Oprire aplicație`.

---

### 4.8 Starea „Oprire aplicație / eliberare resurse”

**Rol:**

* eliberează memoria;
* închide sesiunile GPU;
* finalizează aplicația în mod controlat.

**Tranziție:**
→ `End`

---

## 5. Justificarea utilizării arhitecturii State Machine

Această arhitectură permite:

* control complet asupra fluxului aplicației;
* tratarea explicită a erorilor;
* revenirea sigură într-o stare inițială;
* claritate și simplitate în implementare.

Este o soluție **adecvată din punct de vedere academic** pentru integrarea unei rețele neuronale într-un sistem software funcțional.

---

## 6. Concluzie

Etapa 4 definește arhitectura logică și funcțională a sistemului de detecție a daunelor auto, demonstrând:

* înțelegerea fluxului intern al aplicației;
* integrarea unei rețele neuronale YOLO într-o aplicație reală;
* utilizarea corectă a unei mașini de stări pentru controlul execuției.

Această etapă reprezintă baza pentru **Etapa 5 – antrenarea, evaluarea și validarea performanței modelului**.

---

