"""Orchestrateur : entraîne et compare les 4 modèles, sélectionne le candidat final,
sauvegarde le modèle (pickle), le tableau comparatif et les figures.

Usage : python -m src.train
"""
import json
import pickle
import warnings

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
    """Une seule passe CV : renvoie (probas out-of-fold, AUC par fold)."""
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
        oof, cv_auc = cv_oof_and_auc(model, X_train, y_train, skf)
        print(f"  CV ROC-AUC: {cv_auc.mean():.3f} +/- {cv_auc.std():.3f}", flush=True)

        thr = ev.best_f1_threshold(y_train, oof)
        thresholds[name] = thr
        print(f"  Seuil optimal (max F1, CV train): {thr:.3f}", flush=True)

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
              f"Precision={m['Precision']:.3f} ROC-AUC={m['ROC_AUC']:.3f} PR-AUC={m['PR_AUC']:.3f}",
              flush=True)

    cols = ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC_AUC", "PR_AUC",
            "CV_ROC_AUC_mean", "CV_ROC_AUC_std", "threshold"]
    table = pd.DataFrame(rows)[cols].sort_values("F1", ascending=False).reset_index(drop=True)
    out_csv = config.ROOT / "reports" / "model_comparison.csv"
    table.round(4).to_csv(out_csv, index=False)
    print("\n==================== COMPARAISON ====================")
    print(table.round(3).to_string(index=False))
    print(f"\nTableau -> {out_csv.relative_to(config.ROOT)}")

    best_name = table.iloc[0]["Model"]
    best_model = fitted[best_name]
    best_thr = thresholds[best_name]
    print(f"\n>>> Modele candidat final : {best_name} (seuil={best_thr:.3f})")

    F = config.FIGURES_DIR
    ev.plot_roc_pr(proba_test_all, y_test, F / "10_roc_curves.png", F / "11_pr_curves.png")
    ev.plot_metrics_bar(table, F / "12_metrics_comparison.png")
    ev.plot_confusion(y_test, proba_test_all[best_name], best_thr, best_name,
                      F / "13_confusion_best.png")

    config.MODELS_DIR.mkdir(exist_ok=True)
    with open(config.MODELS_DIR / "best_model.pkl", "wb") as f:
        pickle.dump(best_model, f)
    meta = {
        "model_name": best_name,
        "threshold": best_thr,
        "features": config.ALL_FEATURES,
        "numeric_features": config.NUMERIC_FEATURES,
        "categorical_features": config.CATEGORICAL_FEATURES,
        "target": config.TARGET,
        "metrics_test": table.iloc[0].drop("Model").astype(float).round(4).to_dict(),
    }
    with open(config.MODELS_DIR / "model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print("Modele -> models/best_model.pkl | Metadonnees -> models/model_meta.json")
    print("Figures -> reports/figures/10..13")


if __name__ == "__main__":
    main()
