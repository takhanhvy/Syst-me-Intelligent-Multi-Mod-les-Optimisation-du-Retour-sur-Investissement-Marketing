"""Orchestrateur : entraîne et compare les 4 modèles, sélectionne le candidat final,
sauvegarde le modèle, le tableau comparatif et les figures.

Usage : python -m src.train
"""
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from . import config, evaluation as ev
from .data_preprocessing import load_split
from .modeling import get_models

warnings.filterwarnings("ignore")


def cv_oof_and_auc(model, X_train, y_train, skf):
    """Une seule passe CV : renvoie (probas out-of-fold, AUC par fold).

    Évite de relancer une CV séparée pour la stabilité -> 2x moins d'entraînements.
    """
    X = X_train.reset_index(drop=True)
    y = y_train.reset_index(drop=True)
    oof = np.zeros(len(y))
    fold_aucs = []
    for tr, va in skf.split(X, y):
        m = clone(model)
        m.fit(X.iloc[tr], y.iloc[tr])
        p = m.predict_proba(X.iloc[va])[:, 1]
        oof[va] = p
        fold_aucs.append(roc_auc_score(y.iloc[va], p))
    return oof, np.array(fold_aucs)


def main():
    X_train, X_test, y_train, y_test = load_split()
    print(f"Train: {X_train.shape} | Test: {X_test.shape} | churn train={y_train.mean():.1%}")

    models = get_models(y_train)
    skf = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)

    rows = []
    proba_test_all = {}
    thresholds = {}
    fitted = {}

    for name, model in models.items():
        print(f"\n=== {name} ===", flush=True)
        # Une seule passe CV : probas out-of-fold (pour le seuil) + AUC/fold (stabilité)
        oof, cv_auc = cv_oof_and_auc(model, X_train, y_train, skf)
        print(f"  CV ROC-AUC: {cv_auc.mean():.3f} +/- {cv_auc.std():.3f}", flush=True)

        # Seuil optimal (max F1) calé sur probas out-of-fold du TRAIN
        thr = ev.best_f1_threshold(y_train, oof)
        thresholds[name] = thr
        print(f"  Seuil optimal (max F1, CV train): {thr:.3f}")

        # Entraînement final sur tout le train, évaluation sur le test
        model.fit(X_train, y_train)
        fitted[name] = model
        proba_test = model.predict_proba(X_test)[:, 1]
        proba_test_all[name] = proba_test

        m = ev.metrics_at_threshold(y_test, proba_test, thr)
        m["CV_ROC_AUC_mean"] = cv_auc.mean()
        m["CV_ROC_AUC_std"] = cv_auc.std()
        m["Model"] = name
        rows.append(m)
        print(f"  TEST  F1={m['F1']:.3f} Recall={m['Recall']:.3f} "
              f"Precision={m['Precision']:.3f} ROC-AUC={m['ROC_AUC']:.3f} PR-AUC={m['PR_AUC']:.3f}")

    # Tableau comparatif
    cols = ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC_AUC", "PR_AUC",
            "CV_ROC_AUC_mean", "CV_ROC_AUC_std", "threshold"]
    table = pd.DataFrame(rows)[cols].sort_values("F1", ascending=False).reset_index(drop=True)
    out_csv = config.ROOT / "reports" / "model_comparison.csv"
    table.round(4).to_csv(out_csv, index=False)
    print("\n==================== COMPARAISON ====================")
    print(table.round(3).to_string(index=False))
    print(f"\nTableau -> {out_csv.relative_to(config.ROOT)}")

    # Sélection du candidat final : meilleur F1 (priorité métier sur les churners)
    best_name = table.iloc[0]["Model"]
    best_model = fitted[best_name]
    best_thr = thresholds[best_name]
    print(f"\n>>> Modèle candidat final : {best_name} (seuil={best_thr:.3f})")

    # Figures
    F = config.FIGURES_DIR
    ev.plot_roc_pr(proba_test_all, y_test, F / "10_roc_curves.png", F / "11_pr_curves.png")
    ev.plot_metrics_bar(table, F / "12_metrics_comparison.png")
    ev.plot_confusion(y_test, proba_test_all[best_name], best_thr, best_name,
                      F / "13_confusion_best.png")

    # Sauvegarde du modèle final + métadonnées
    config.MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(best_model, config.MODELS_DIR / "best_model.joblib")
