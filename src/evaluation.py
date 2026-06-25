"""Évaluation et comparaison des modèles.

Méthodo anti-leakage pour le seuil :
- Le seuil de décision optimal (max F1) est calé sur des probabilités obtenues par
  cross_val_predict SUR LE TRAIN uniquement, puis appliqué tel quel au test.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from . import config


def best_f1_threshold(y_true, proba):
    """Seuil maximisant le F1 (sur des probabilités de validation)."""
    prec, rec, thr = precision_recall_curve(y_true, proba)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    # thr a une longueur len(prec)-1
    idx = int(np.nanargmax(f1[:-1])) if len(thr) else 0
    return float(thr[idx]) if len(thr) else 0.5


def cv_oof_proba(model, X_train, y_train):
    """Probabilités out-of-fold sur le train (pour seuil + stabilité), sans fuite."""
    skf = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    proba = cross_val_predict(model, X_train, y_train, cv=skf, method="predict_proba", n_jobs=-1)
    return proba[:, 1]


def metrics_at_threshold(y_true, proba, thr):
    pred = (proba >= thr).astype(int)
    return {
        "Accuracy": accuracy_score(y_true, pred),
        "Precision": precision_score(y_true, pred, zero_division=0),
        "Recall": recall_score(y_true, pred, zero_division=0),
        "F1": f1_score(y_true, pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_true, proba),
        "PR_AUC": average_precision_score(y_true, proba),
        "threshold": thr,
    }


def plot_roc_pr(results, y_test, path_roc, path_pr):
    """results: dict {name: proba_test}."""
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, proba in results.items():
        fpr, tpr, _ = roc_curve(y_test, proba)
        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_test, proba):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("Taux de faux positifs"); ax.set_ylabel("Taux de vrais positifs")
    ax.set_title("Courbes ROC — comparaison des modèles"); ax.legend()
    fig.tight_layout(); fig.savefig(path_roc, dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    base = y_test.mean()
    for name, proba in results.items():
        prec, rec, _ = precision_recall_curve(y_test, proba)
        ax.plot(rec, prec, label=f"{name} (AP={average_precision_score(y_test, proba):.3f})")
    ax.axhline(base, color="k", ls="--", lw=0.8, label=f"hasard ({base:.2f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Courbes Precision-Recall — comparaison des modèles"); ax.legend()
    fig.tight_layout(); fig.savefig(path_pr, dpi=120); plt.close(fig)


def plot_confusion(y_test, proba, thr, name, path):
    pred = (proba >= thr).astype(int)
    cm = confusion_matrix(y_test, pred)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center",
                color="white" if v > cm.max() / 2 else "black")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Reste", "Churn"]); ax.set_yticklabels(["Reste", "Churn"])
    ax.set_xlabel("Prédit"); ax.set_ylabel("Réel")
    ax.set_title(f"Matrice de confusion — {name}\n(seuil={thr:.2f})")
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def plot_metrics_bar(table, path):
    cols = ["Recall", "F1", "ROC_AUC", "PR_AUC"]
    ax = table.set_index("Model")[cols].plot(kind="bar", figsize=(10, 5))
    ax.set_title("Comparaison des modèles (métriques clés sur le test)")
    ax.set_ylabel("Score"); ax.tick_params(axis="x", rotation=0)
    ax.legend(loc="lower right")
    ax.figure.tight_layout(); ax.figure.savefig(path, dpi=120); plt.close(ax.figure)
