
---

# 📘 Etapa 3 – Analiza și Pregătirea Setului de Date pentru Rețele Neuronale

**Disciplina:** Rețele Neuronale
**Instituție:** POLITEHNICA București – FIIR
**Student:** Baba Cristian Teodor
**Proiect:** Detecția daunelor auto utilizând rețele neuronale
**An universitar:** 2025–2026

---

## 1. Introducere

Această etapă are ca scop **analiza și pregătirea setului de date** utilizat în proiectul de rețele neuronale. Calitatea și organizarea corectă a datelor reprezintă un factor esențial pentru obținerea unor rezultate relevante în etapele ulterioare de antrenare și evaluare a modelului.

În cadrul acestui proiect, datele sunt reprezentate de **imagini reale cu daune auto**, împreună cu adnotări în format YOLO, iar preprocesarea este intenționat menținută **simplă și controlată**, fiind aplicată **o singură operație: ENHANCE**.

---

## 2. Descrierea setului de date

### 2.1 Tipul datelor

* **Date de intrare:** imagini RGB (`.jpg`, `.png`)
* **Adnotări:** fișiere `.txt` în format YOLO (bounding boxes normalizate)
* **Tip problemă:** detecție de obiecte (*object detection*)

### 2.2 Clasele definite

Setul de date conține următoarele clase de daune auto:

| ID | Clasă         |
| -- | ------------- |
| 0  | dent          |
| 1  | scratch       |
| 2  | crack         |
| 3  | glass_shatter |
| 4  | lamp_broken   |
| 5  | tire_flat     |

---

## 3. Structura reală a datasetului

Structura prezentată mai jos reflectă **exact organizarea folderelor din proiect**, fără etape sau directoare suplimentare.

```
project-root/
│
├── config/
│   └── car_damage.yaml
│
├── data/
│   ├── images/
│   │   ├── manual/
│   │   │   ├── images/
│   │   │   └── labels/
│   │   │
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
```

### Observații importante

* Folderul `manual` conține **contribuția proprie**, păstrată separat pentru trasabilitate.
* Datasetul din `train / val / test` reprezintă setul final utilizat pentru antrenare.
* `images_enhanced` este o **copie preprocesată vizual** a datasetului original.

---

## 4. Analiza exploratorie a datelor (EDA)

Înainte de preprocesare, au fost realizate următoarele verificări:

* existența unui fișier `.txt` pentru fiecare imagine;
* validarea formatului YOLO (ID clasă + coordonate normalizate);
* identificarea imaginilor fără adnotări valide;
* analiza distribuției claselor pentru detectarea dezechilibrelor.

### Probleme observate

* unele clase (ex. *crack*, *scratch*) sunt mai dificil de detectat din cauza contrastului redus;
* există variații mari de iluminare și reflexii ale caroseriei;
* distribuția claselor nu este perfect uniformă.

---

## 5. Preprocesarea datelor – ENHANCE

### 5.1 Operația aplicată

În cadrul proiectului a fost aplicată **o singură etapă de preprocesare**, denumită **ENHANCE**, care constă în:

* îmbunătățirea contrastului local;
* evidențierea detaliilor fine (zgârieturi, crăpături);
* reducerea efectelor iluminării neuniforme.

### 5.2 Ce NU se aplică

Pentru a menține fidelitatea datelor:

* nu se aplică redimensionări,
* nu se aplică rotații sau flip-uri,
* nu se aplică augmentări artificiale,
* etichetele YOLO rămân neschimbate.

### 5.3 Motivația alegerii

Această abordare a fost aleasă pentru:

* a evita introducerea artefactelor artificiale;
* a păstra realismul daunelor auto;
* a asigura un pipeline simplu și ușor de justificat academic.

---

## 6. Împărțirea datasetului

Datasetul a fost împărțit respectând cerința:

|     Subset | Procent |
| ---------: | :------ |
|      Train | 70%     |
| Validation | 15%     |
|       Test | 15%     |

Principii respectate:

* seturi disjuncte (fără suprapuneri);
* fără *data leakage*;
* seturile de validare și test sunt utilizate exclusiv pentru evaluare.

---

## 7. Fișiere rezultate în Etapa 3

* dataset organizat în `data/images/`;
* versiune preprocesată în `data/images_enhanced/`;
* fișiere YOLO validate;
* fișier de configurare `car_damage.yaml`.

---

## 8. Concluzie

În urma Etapei 3, setul de date este:

* corect organizat pentru YOLO;
* validat din punct de vedere al etichetelor;
* îmbunătățit vizual prin operația ENHANCE;
* pregătit pentru **Etapa 4 – definirea arhitecturii rețelei neuronale** și **Etapa 5 – antrenarea modelului**.

---

