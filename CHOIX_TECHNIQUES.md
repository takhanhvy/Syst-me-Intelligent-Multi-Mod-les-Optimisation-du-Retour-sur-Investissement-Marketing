# Choix techniques — Système de rétention client (prédiction du churn)

Document de cadrage validé avant implémentation. Référence pour le code, le rapport et la soutenance (RNCP40875 — Bloc 2).

## 1. Tâche prédictive

| Tâche | Type | Cible |
|-------|------|-------|
| Prédiction du churn | Classification binaire | `churn` (0 = reste, 1 = résilie) |

Scope volontairement resserré sur le churn (tâche obligatoire, exigences pleinement couvertes). Le revenu à risque pourra être ajouté plus tard comme simple **KPI business** dans le dashboard (`proba_churn × total_revenue`), sans modèle dédié.

## 2. Dataset

- `data/customer_churn_business_dataset.csv` — **10 000 clients × 32 variables**
- **Déséquilibre : churn = 10,2 %** (classe positive minoritaire) → métriques et stratégies adaptées (cf. §5)
- Manquants : `complaint_type` (2045) → imputé `"None"` (= pas de plainte)
- `customer_id` retiré (identifiant, non prédictif)

## 3. Stack technique

- **Langage** : Python 3.11
- **Data / ML** : pandas, NumPy, scikit-learn
- **Gradient Boosting** : XGBoost
- **Deep Learning** : `sklearn.neural_network.MLPClassifier` (intégré aux pipelines sklearn)
- **Explicabilité** : `shap` + `permutation_importance`
- **Dashboard** : Streamlit + Plotly
- **API** : FastAPI + Uvicorn (validation Pydantic)
- **Sérialisation** : joblib · **Versioning** : Git (branche `dev_vy`)

> Le `MLPClassifier` sklearn satisfait l'exigence « ≥ 1 modèle Deep Learning ». Bascule possible vers Keras plus tard (même interface pipeline) si on veut renforcer ce point pour la note.

## 4. Modèles comparés (≥ 4)

1. **Régression logistique** — *baseline*, interprétable
2. **Random Forest** — non-linéarités
3. **XGBoost** — performance
4. **MLPClassifier** — Deep Learning

## 5. Préparation des données (anti-leakage)

- Pipeline **scikit-learn** : `ColumnTransformer` + `Pipeline`
- Numériques : imputation médiane + `StandardScaler`
- Catégorielles : imputation mode/`"None"` + `OneHotEncoder(handle_unknown="ignore")`
- **Preprocessing fit sur le train uniquement**, appliqué au test
- Split stratifié `train_test_split(stratify=churn)` + **StratifiedKFold** (cross-validation)
- Déséquilibre : `class_weight="balanced"` / `scale_pos_weight` (XGBoost) + **ajustement du seuil de décision**

## 6. Évaluation

- **Métriques** : Accuracy, Precision, Recall, F1, ROC-AUC, **PR-AUC** + matrice de confusion
- **Métrique de décision priorisée : Recall / F1** (coût élevé des faux négatifs = churners manqués)
- Tableaux comparatifs + courbes ROC/PR ; **modèle candidat final** argumenté (performance / stabilité / interprétabilité / coût / déploiement)
- Analyse d'erreurs (matrice de confusion, cas mal classés)

## 7. Explicabilité

- `feature_importances_` (arbres) + **permutation importance** (agnostique)
- **SHAP** sur le modèle final : importance globale + explication locale d'un client

## 8. Dashboard (Streamlit) — orienté métier

KPI (nb de clients à risque, revenu global à risque), saisie d'un scénario client → proba de churn en temps réel, simulation « what-if », comparaison des modèles, top facteurs (SHAP). **Appelle l'API** pour les prédictions.

## 9. API REST (FastAPI)

- `POST /predict` — JSON features → classe + probabilité de churn
- `GET /health` — état du service + modèle chargé
- `GET /model-info` — métadonnées (optionnel)
- Validation Pydantic + gestion d'erreurs + codes HTTP + Swagger auto

## 10. Structure cible du repo

```
.
├── data/
│   └── customer_churn_business_dataset.csv
├── eda/
│   └── 01_eda.ipynb              # EDA & expérimentation
├── src/
│   ├── config.py                 # chemins, constantes, listes de features
│   ├── data_preprocessing.py     # pipelines sklearn
│   ├── modeling.py               # définition + entraînement des modèles
│   ├── evaluation.py             # métriques, comparaison, graphes
│   ├── explainability.py         # SHAP / permutation importance
│   └── train.py                  # orchestrateur → artefacts joblib
├── api/main.py                   # FastAPI
├── dashboard/app.py              # Streamlit
├── models/                       # modèles sérialisés (.joblib)
├── reports/figures/              # visuels EDA & évaluation
├── requirements.txt
├── README.md
└── .gitignore
```

Pipeline final exécutable **hors notebook** (`python -m src.train`). Commits Git réguliers et explicites.

---

### Décisions verrouillées
Tâche : **Churn uniquement (classification binaire)** · DL : **sklearn MLPClassifier** · API : **FastAPI** · Structure : **modules `src/` + notebook EDA** · Métrique pilote : **F1 / Recall** (déséquilibre 10 %).
