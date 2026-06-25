# Système Intelligent de Rétention Client — Prédiction du Churn

Plateforme complète d'aide à la décision pour anticiper la résiliation client (churn) et
en évaluer l'impact financier, de la donnée brute jusqu'à la mise à disposition opérationnelle
(API + dashboard).

> Projet Data Science — EFREI M1 Data Engineering & AI — Épreuve certifiante **RNCP40875, Bloc 2**
> (Piloter et implémenter des solutions d'IA).

---

## 1. Objectif

À partir de données comportementales de 10 000 clients (usage, facturation, support,
satisfaction), le système prédit la **probabilité de churn** (classification binaire) et la
transforme en outil de pilotage : KPIs, priorisation des clients à risque, revenu à risque,
simulation de scénarios et explications actionnables.

**Tâche retenue :** classification binaire `churn` (0 = reste, 1 = résilie).

## 2. Données

- Source : [Customer Churn Prediction Business Dataset (Kaggle)](https://www.kaggle.com/datasets/miadul/customer-churn-prediction-business-dataset)
- `data/customer_churn_business_dataset.csv` — **10 000 clients × 32 variables**
- Cible déséquilibrée : **~10 % de churn** → métriques adaptées (Recall / F1 / PR-AUC),
  `class_weight` / `scale_pos_weight`, et ajustement du seuil de décision.

## 3. Architecture

Architecture réaliste **Front / API / Modèle** : le dashboard n'embarque pas le modèle, il
interroge l'API REST, qui charge le pipeline sérialisé.

```
Dashboard (Streamlit)  ──HTTP──►  API REST (FastAPI)  ──►  best_model.pkl (pipeline sklearn)
   KPIs / simulateur                /predict /predict-batch        + model_meta.json (seuil, métriques)
                                    /explain /model-info /health
```

## 4. Structure du dépôt

```
.
├── data/                          # dataset brut
├── eda/01_eda.ipynb               # analyse exploratoire (notebook exécuté)
├── src/
│   ├── config.py                  # chemins, constantes, listes de features
│   ├── data_preprocessing.py      # ColumnTransformer anti-leakage + split
│   ├── modeling.py                # définition des 4 modèles
│   ├── evaluation.py              # métriques, seuil, courbes ROC/PR
│   ├── explainability.py          # permutation importance + SHAP
│   └── train.py                   # orchestrateur -> modèle + métriques + figures
├── api/
│   ├── main.py                    # API FastAPI
│   └── test_api.py                # tests (10) des endpoints
├── dashboard/app.py               # interface décisionnelle métier (Streamlit)
├── models/                        # best_model.pkl (ignoré par git) + model_meta.json
├── reports/
│   ├── figures/                   # figures EDA, comparaison, SHAP
│   └── model_comparison.csv       # tableau comparatif des modèles
├── CHOIX_TECHNIQUES.md            # cadrage et justification des choix
├── requirements.txt
└── README.md
```

## 5. Installation

```bash
python -m venv .venv
# Windows :  .venv\Scripts\activate      |  macOS/Linux :  source .venv/bin/activate
python -m pip install -r requirements.txt
```

> **Windows — chemins longs.** Si l'installation échoue avec
> `OSError ... enable Windows Long Path support`, c'est la limite des 260 caractères.
> Solutions : activer les *Long Paths* (admin), **ou** créer le venv sur un chemin court
> (`python -m venv C:\cv` puis l'activer), **ou** raccourcir le dossier du projet.

## 6. Utilisation

**Entraîner et comparer les modèles** (génère `models/best_model.pkl`, `model_meta.json`,
les figures et `reports/model_comparison.csv`) :

```bash
python -m src.train
```

**Lancer l'API** (puis doc interactive sur http://127.0.0.1:8000/docs) :

```bash
uvicorn api.main:app --reload
```

**Lancer le dashboard** (API démarrée en parallèle) :

```bash
streamlit run dashboard/app.py
```

**Explicabilité** (permutation importance + SHAP) et **tests API** :

```bash
python -m src.explainability
python -m api.test_api
```

### Endpoints de l'API

| Méthode | Endpoint         | Rôle |
|--------|-------------------|------|
| GET    | `/health`         | État du service et du modèle |
| GET    | `/model-info`     | Nom du modèle, seuil, métriques, facteurs de churn |
| POST   | `/predict`        | Prédiction pour un client |
| POST   | `/predict-batch`  | Prédiction pour N clients (KPIs dashboard) |
| POST   | `/explain`        | Facteurs SHAP locaux d'un client |

## 7. Démarche & résultats

- **Préparation anti-leakage** : tout le preprocessing (imputation, scaling, encodage) est
  encapsulé dans un `Pipeline` sklearn, *fit* uniquement sur le train à chaque fold.
- **4 modèles comparés** : Régression logistique (baseline), Random Forest, XGBoost, et un
  réseau de neurones **MLP** (Deep Learning). Comparaison par validation croisée + test.
- **Seuil de décision** optimisé (max F1) sur des probabilités *out-of-fold* du train, donc
  sans fuite vers le test.
- **Modèle candidat final** : sélectionné sur le meilleur F1 (priorité métier aux churners),
  en tenant compte de la stabilité (écart-type CV), de l'interprétabilité et du coût.
  Random Forest et XGBoost sont au coude-à-coude en tête ; le MLP est le plus instable,
  illustrant que « le Deep Learning n'est pas toujours supérieur ».
- **Explicabilité** : permutation importance (globale) et SHAP (global + local). Les
  facteurs dominants sont le **CSAT**, le **nombre de connexions**, l'**ancienneté** et les
  **échecs de paiement** — relations à effet de seuil, ce qui avantage les modèles non
  linéaires sur la régression logistique.

Le détail des choix est documenté dans [`CHOIX_TECHNIQUES.md`](CHOIX_TECHNIQUES.md).

## 8. Distinction analyse vs dashboard

Les visualisations scientifiques (EDA, courbes ROC/PR, matrices de confusion, comparaison
des modèles) appuient la **démarche** et restent dans le rapport / `reports/figures/`.
Le **dashboard** est un outil opérationnel orienté décideur (marketing / CRM / finance) :
KPIs, revenu à risque, priorisation, simulation et explications — sans graphiques scientifiques.

## 9. Stack

Python · pandas · scikit-learn · XGBoost · SHAP · FastAPI · Streamlit · Plotly
