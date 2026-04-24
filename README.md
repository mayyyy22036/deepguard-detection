<img width="2600" height="1330" alt="gradcam_visualization" src="https://github.com/user-attachments/assets/10dd7779-50de-4808-816a-b3b7e8dadedf" /># 🛡️ DeepGuard — Système de Détection de Deepfakes par Deep Learning

> **Module** : Deep Learning — Computer Vision & Modèles Génératifs  
> **Encadrant** : Haythem Ghazouani  
> **Période** : Mars – Avril 2026  
> **Niveau** : Terminale Data Science

---

## 📊 Résultats Finaux

| Métrique | Valeur | Interprétation |
|----------|--------|----------------|
| **Accuracy** | 81.29% | Images correctement classifiées |
| **AUC-ROC** | **0.9310** | Capacité de discrimination (1.0 = parfait) |
| **F1-Score** | 0.8365 | Équilibre précision / rappel |
| **Recall** | **0.9694** | 97% des deepfakes détectés |
| **Precision** | 0.6842 | Fiabilité des alertes FAKE |

> L'AUC-ROC est la métrique principale : 0.93 signifie que le modèle distingue correctement un deepfake d'une image réelle dans 93% des cas, indépendamment du seuil de décision. Ce résultat est comparable à l'état de l'art sur la compression c40 de FaceForensics++.

---

## 🎯 Contexte et Motivation

En 2026, les deepfakes générés par des modèles comme Sora ou HeyGen sont devenus indiscernables à l'œil nu. Cette technologie représente des menaces concrètes :

- **Désinformation** : fausses vidéos de personnalités publiques
- **Fraude identitaire** : usurpation dans les appels vidéo
- **Contenu non-consensuel** : manipulation d'images de personnes

DeepGuard est un système complet de détection de bout en bout : il prend une image en entrée et retourne une prédiction REAL/FAKE accompagnée d'un score de confiance et d'une visualisation explicative via Grad-CAM.

---

## 🏗️ Architecture Technique

### Pipeline global

```
Vidéos brutes (MP4)
        ↓
Extraction de frames (32 par vidéo)
        ↓
Détection de visages — MTCNN
(crop + marge 20%, resize 299×299)
        ↓
Split par VIDEO ID (anti data leakage)
Train 70% | Val 15% | Test 15%
        ↓
┌──────────────────────────────────────┐
│   Xception (pré-entraîné ImageNet)   │
│   Backbone depthwise separable conv  │
│   Features : 2048 dimensions         │
│   Head : Dropout(0.5) → Linear(2048,2)│
│   Input : 299×299 RGB                │
│   Normalisation : [-1, 1]            │
└──────────────────────────────────────┘
        ↓
Grad-CAM — Explainabilité visuelle
        ↓
FastAPI Backend (/predict endpoint)
        ↓
React Frontend (interface interactive)
```

### Pourquoi Xception ?

Xception (Chollet, CVPR 2017) est l'architecture de référence pour la détection de deepfakes. Ses **depthwise separable convolutions** capturent séparément les corrélations spatiales et inter-canaux, ce qui est particulièrement adapté pour détecter les artefacts subtils laissés par les algorithmes de génération de deepfakes.

Rossler et al. (2019) dans le paper FaceForensics++ confirment : *"XceptionNet outperforms all baselines"*.

### Comparaison des architectures testées

| Architecture | Paramètres | Val AUC | Décision |
|---|---|---|---|
| EfficientNet-B0 | 5.3M | 0.84 | Baseline initial — trop petit |
| EfficientNet-B4 | 19M | 0.82 | Overfitting sur ~5000 images |
| **Xception** | **22.9M** | **0.93** | ✅ Retenu — optimal pour deepfakes |

---

## 📦 Datasets

### FaceForensics++ (FF++) — c40

- **Source** : Université Technique de Munich — accès académique (~24h)
- **Contenu** : 50 vidéos originales YouTube + 50 vidéos manipulées (Deepfakes)
- **Compression** : c40 (~10 GB vs 500 GB pour c23) — choix justifié par les contraintes de stockage
- **Techniques** : Deepfakes, Face2Face, FaceSwap, NeuralTextures

### Celeb-DF v2

- **Source** : Université de l'État de New York
- **Contenu** : Celeb-real + YouTube-real (réelles) + Celeb-synthesis (fakes haute qualité)
- **Utilisé** : 200 vidéos réelles + 200 vidéos fake (équilibrage volontaire)

### Statistiques du dataset final

```
Source        | Réelles | Fakes   | Frames/vidéo
--------------|---------|---------|-------------
FF++ c40      | 50 vid  | 50 vid  | 32
Celeb-DF v2   | 200 vid | 200 vid | 16
--------------|---------|---------|-------------
Total faces   | ~4 800  | ~4 800  | ~9 600 total

Split (par VIDEO ID) :
  Train : 70% — ~6 720 images
  Val   : 15% — ~1 440 images
  Test  : 15% — ~1 440 images
```

### Problème critique résolu : Data Leakage

Un split naïf par frames donnait une accuracy artificielle de 100% — le modèle mémorisait les visages au lieu d'apprendre à détecter les deepfakes.

```python
# ❌ INCORRECT — data leakage
train_test_split(all_frames, test_size=0.3)
# Frames de la même vidéo dans train ET test → mémorisation des visages

# ✅ CORRECT — notre approche
train_test_split(video_ids, test_size=0.3)
# Toutes les frames d'une vidéo restent dans le même split
```

**Impact** : drop de 100% → 81-89% accuracy réaliste, confirmant la vraie généralisation du modèle.

---

## 🔬 Entraînement

### Preprocessing

1. Extraction de 32 frames uniformément espacées par vidéo (16 pour Celeb-DF)
2. Détection de visage via MTCNN avec marge de 20% autour du visage détecté
3. Redimensionnement à 299×299 pixels (taille native Xception)
4. Normalisation dans [-1, 1] (spécifique à Xception, différent d'ImageNet)

### Data Augmentation (training uniquement)

```python
A.HorizontalFlip(p=0.5)
A.Rotate(limit=10, p=0.5)
A.RandomBrightnessContrast(0.2, 0.2, p=0.5)
A.GaussianBlur(blur_limit=(3, 5), p=0.3)
A.ImageCompression(quality_lower=60, p=0.3)   # simule artefacts JPEG
```

### Hyperparamètres

| Paramètre | Valeur | Justification |
|-----------|--------|---------------|
| LR backbone | 5e-5 | Fine-tuning lent — préserver ImageNet |
| LR head | 5e-4 | Apprendre depuis zéro |
| Batch Size | 16 | Xception lourd (22M params) |
| Dropout | 0.5 | Prévention overfitting |
| Label Smoothing | 0.1 | Meilleure généralisation |
| Scheduler | CosineAnnealingLR | Descente progressive du LR |
| Early Stopping | patience=7 | Arrêt si AUC stagne |
| Optimizer | AdamW | LR adaptatif + weight decay |

### Differential Learning Rates

Le backbone pré-entraîné apprend 10× plus lentement que la tête de classification — technique clé pour préserver les features visuelles d'ImageNet tout en adaptant le modèle aux deepfakes.

```python
optimizer = AdamW([
    {'params': model.backbone.parameters(), 'lr': 5e-5},  # backbone — lent
    {'params': model.classifier.parameters(), 'lr': 5e-4}, # head — rapide
])
```

### Plateforme d'entraînement

- **GPU** : Kaggle GPU T4 (15 GB VRAM) — 2-3h pour 20 epochs
- **Tracking** : MLflow (métriques par epoch, checkpoints)
- **CPU local** : également testé (référence baseline)

### Ablation Study (Semaine 3)

6 configurations testées sur EfficientNet-B0 pour valider les hyperparamètres :

| Configuration | Val Acc | Δ Baseline |
|---|---|---|
| Baseline (retenu) | 89.06% | — |
| Sans dropout | 83.93% | -5.13% |
| Dropout 0.5 | 87.50% | -1.56% |
| LR × 10 | 85.49% | -3.57% |
| MixUp | 85.94% | -3.12% |
| Batch 64 | 84.38% | -4.69% |

---

## 🔍 Explainabilité — Grad-CAM

### Principe

Grad-CAM (Selvaraju et al., ICCV 2017) visualise les zones de l'image qui ont influencé la décision du modèle. Pour la détection de deepfakes, cela permet de vérifier que le modèle regarde les bonnes régions : contours du visage, yeux, bouche — là où les algorithmes de deepfake laissent des artefacts de fusion.

### Implémentation

```python
# Forward pass → prédiction
# Backward pass → gradients sur le dernier feature map
# Weighted average → heatmap normalisée
# Superposition sur l'image originale (alpha=0.45)
```

### Observations



- **Images REAL** : Les activations sont centrées sur les zones clés du visage (nez, bouche) avec une intensité modérée. Le modèle analyse des features globales sans détecter d’anomalies évidentes.
- **Images FAKE** : Les activations sont plus intenses et concentrées sur les régions centrales du visage, notamment le nez et la bouche, suggérant que le modèle capte des incohérences subtiles dans ces zones (artefacts de génération ou de fusion).

---

## 🚀 Installation et Lancement

### Prérequis

```
Python 3.11
Node.js 18+
Docker + Docker Compose (optionnel)
```

### Backend (FastAPI)

```bash
cd backend
pip install -r requirements.txt
python main.py
# API : http://localhost:8000
# Documentation Swagger : http://localhost:8000/docs
```

### Frontend (React)

```bash
cd frontend
npm install
npm run dev
# Interface : http://localhost:3000
```

### Docker (déploiement complet)

```bash
docker-compose up -d
# Backend  : http://localhost:8000
# Frontend : http://localhost:3000
```

---

## 📡 API Reference

### `POST /predict`

Analyse une image et retourne la prédiction deepfake.

**Request :**
```
Content-Type: multipart/form-data
Body: file (jpg, jpeg, png)
```

**Response :**
```json
{
  "prediction": "Fake",
  "confidence": 94.32,
  "probabilities": {
    "real": 5.68,
    "fake": 94.32
  },
  "processing_time": 0.38
}
```

### `GET /health`

```json
{
  "model_exists": true,
  "model_path": "models/deepguard_xception.pth"
}
```

### `GET /`

```json
{
  "status": "ok",
  "device": "cpu"
}
```

---

## 📁 Structure du Projet

```
deepguard-detection/
│
├── backend/                        # API FastAPI
│   ├── main.py                     # Application principale + endpoints
│   ├── requirements.txt
│   └── Dockerfile
│
├── frontend/                       # Interface React + TypeScript
│   ├── src/
│   │   ├── pages/
│   │   │   ├── Home.tsx
│   │   │   ├── Detection.tsx       # Upload + résultats
│   │   │   └── About.tsx
│   │   ├── components/
│   │   │   ├── FileUpload.tsx
│   │   │   ├── ResultDisplay.tsx
│   │   │   └── LoadingSpinner.tsx
│   │   └── api/client.ts           # Axios + appels API
│   ├── package.json
│   └── Dockerfile
│
├── src/                            # Pipeline ML
│   ├── data/
│   │   ├── dataset.py              # PyTorch Dataset + DataLoaders
│   │   ├── preprocessing.py        # MTCNN + extraction de visages
│   │   └── augmentation.py         # MixUp (testé, non retenu)
│   ├── models/
│   │   ├── architecture.py         # DeepfakeClassifier (Xception, B0, B4)
│   │   └── training.py             # Trainer + MLflow tracking
│   └── explainability/
│       └── gradcam.py              # Grad-CAM implementation
│
├── scripts/
│   ├── preprocess_corrected.py     # Split par VIDEO ID (anti data leakage)
│   ├── train.py                    # Script d'entraînement local
│   └── ablation_study.py           # 6 configurations testées (Semaine 3)
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_dataset_test.ipynb
│   ├── 03_model_evaluation.ipynb
│   ├── 04_week3_report.ipynb
│   └── 05_gradcam_visualization.ipynb
│
├── models/                         # Gitignored
│   └── deepguard_xception.pth      # Modèle final
│
├── results/
│   └── gradcam_visualization.png
│
├── mlruns/                         # MLflow tracking
├── docker-compose.yml
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🔍 Défis Techniques et Solutions

### 1. Data Leakage (Semaine 1)
**Problème** : Split par frames → 100% accuracy artificielle  
**Cause** : Frames de la même vidéo dans train et test → mémorisation des visages  
**Solution** : Split par VIDEO ID — toutes les frames d'une vidéo dans le même split  
**Impact** : Performance réaliste 81-89%

### 2. Overfitting (Semaine 2-3)
**Problème** : Train accuracy 94% vs Val accuracy 83% (écart de 11%)  
**Solution** : Dropout 0.5, label smoothing 0.1, augmentation agressive  
**Leçon** : EfficientNet-B4 trop grand pour ~5000 images → Xception plus adapté

### 3. Conflits de dépendances (Kaggle)
**Problème** : ImportError Pillow 12.x incompatible avec facenet-pytorch  
**Solution** : Ne pas upgrader Pillow, installer facenet-pytorch séparément  
**Leçon** : Fixer les versions dans requirements.txt dès le début

### 4. Équilibrage du dataset
**Problème** : 2400 real vs 4000 fake après fusion FF++ + Celeb-DF  
**Solution** : Sous-échantillonnage aléatoire des fakes à 2400  
**Raison** : Éviter le biais du modèle vers la classe majoritaire

---

## 📈 Progression Semaine par Semaine

| Semaine | Objectif | Réalisé | Résultat clé |
|---------|----------|---------|--------------|
| 1 | Data Pipeline | ✅ MTCNN + split VIDEO ID | 3 200 faces extraites, data leakage corrigé |
| 2 | Baseline | ✅ EfficientNet-B0 + MLflow | AUC = 0.86 (sur données propres) |
| 3 | Optimisation | ✅ Ablation study (6 configs) | Hyperparamètres baseline validés |
| 4 | Explainabilité | ✅ Grad-CAM | Visualisations sur images REAL/FAKE |
| 5 | Déploiement | ✅ FastAPI + React + Docker | Endpoint /predict fonctionnel |
| + | Upgrade | ✅ Xception + Celeb-DF v2 | AUC = 0.93 sur Xception |

---

## 🛠️ Stack Technique

**Machine Learning**
- PyTorch 2.x · timm · albumentations · facenet-pytorch · scikit-learn

**MLOps**
- MLflow (experiment tracking) · Kaggle GPU T4 (entraînement)

**Backend**
- FastAPI · Uvicorn · OpenCV · Pillow · albumentations

**Frontend**
- React 18 · TypeScript · Tailwind CSS · Axios · Vite

**Infrastructure**
- Docker · Docker Compose · GitHub

---

## ⚖️ Considérations Éthiques et Limites

### Limites du modèle

Le modèle est entraîné sur des **face swaps et face reenactments** (FaceForensics++, Celeb-DF). Il sera moins efficace sur :
- Des images entièrement générées par text-to-image (Midjourney, DALL-E, Stable Diffusion)
- Des deepfakes issus de modèles de diffusion récents (Sora, Runway)

Ces cas représentent une direction de recherche future : inclure des datasets de diffusion models.

### Biais potentiels

- Le dataset FF++ est majoritairement composé de visages de personnes à peau claire
- Les performances peuvent varier selon l'origine ethnique et l'âge des sujets
- La compression c40 rend la détection plus difficile que sur les vidéos haute qualité

### Usage responsable

Ce système est conçu à des fins de recherche et de sensibilisation. Il ne doit pas être utilisé pour surveiller des individus à leur insu ni pour prendre des décisions automatisées sans vérification humaine.

---

## 📚 Références

1. Rossler, A. et al. (2019). **FaceForensics++: Learning to Detect Manipulated Facial Images**. *ICCV 2019*.
2. Chollet, F. (2017). **Xception: Deep Learning with Depthwise Separable Convolutions**. *CVPR 2017*.
3. Li, Y. et al. (2020). **Celeb-DF: A Large-scale Challenging Dataset for DeepFake Forensics**. *CVPR 2020*.
4. Selvaraju, R. et al. (2017). **Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization**. *ICCV 2017*.
5. Tan, M., Le, Q. (2019). **EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks**. *ICML 2019*.

---

## 🔗 Liens

- **Repository** : https://github.com/mayyyy22036/deepguard-detection
- **API Swagger** : http://localhost:8000/docs (en local)
- **MLflow UI** : http://localhost:5000 (en local)

---

*DeepGuard — Projet Deep Learning — Mars-Avril 2026*  
*Encadrant : Haythem Ghazouani — h.ghazouani@pi.tn*
