"""Explicabilité du modèle final (Random Forest).

- Permutation importance : agnostique au modèle, sur le test set (scoring ROC-AUC).
- SHAP (TreeExplainer) : importance globale (summary) + explication locale d'un client.

Usage : python -m src.explainability
"""
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from . import config
from .data_preprocessing import load_split, get_feature_names


def load_model():
    with open(config.MODELS_DIR / "best_model.pkl", "rb") as f:
        return pickle.load(f)


def permutation_fig(model, X_test, y_test, path, n_repeats=5, top=20):
    """Permutation importance sur le pipeline complet (scoring ROC-AUC)."""
    r = permutation_importance(
        model, X_test, y_test, scoring="roc_auc",
        n_repeats=n_repeats, random_state=config.RANDOM_STATE, n_jobs=-1,
    )
    imp = pd.Series(r.importances_mean, index=config.ALL_FEATURES).sort_values()
    imp = imp.tail(top)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(imp.index, imp.values, color="#4C72B0",
            xerr=r.importances_std[[config.ALL_FEATURES.index(i) for i in imp.index]])
    ax.set_title("Permutation importance (chute de ROC-AUC) — top variables")
    ax.set_xlabel("Baisse moyenne de ROC-AUC quand la variable est mélangée")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    return imp.sort_values(ascending=False)


def _transform(model, X):
    prep = model.named_steps["prep"]
    Xt = prep.transform(X)
    if hasattr(Xt, "toarray"):
        Xt = Xt.toarray()
    names = get_feature_names(prep)
    return pd.DataFrame(Xt, columns=names, index=X.index), names


def shap_global_fig(model, X_sample, path):
    import shap
    rf = model.named_steps["model"]
    Xt_df, names = _transform(model, X_sample)
    explainer = shap.TreeExplainer(rf)
    sv = explainer.shap_values(Xt_df, check_additivity=False)
    sv1 = sv[1] if isinstance(sv, list) else (sv[:, :, 1] if getattr(sv, "ndim", 2) == 3 else sv)
    plt.figure()
    shap.summary_plot(sv1, Xt_df, max_display=20, show=False)
    plt.title("SHAP — importance globale (impact sur la proba de churn)")
    plt.tight_layout(); plt.savefig(path, dpi=120, bbox_inches="tight"); plt.close()
    return explainer, names


def shap_local_fig(model, explainer, X_client, names, path):
    """Top contributions SHAP pour un client (waterfall manuel, robuste)."""
    import shap
    rf = model.named_steps["model"]
    Xt_df, _ = _transform(model, X_client)
    sv = explainer.shap_values(Xt_df, check_additivity=False)
    sv1 = sv[1] if isinstance(sv, list) else (sv[:, :, 1] if getattr(sv, "ndim", 2) == 3 else sv)
    contrib = pd.Series(np.asarray(sv1)[0], index=names)
    top = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(12)[::-1]
    colors = ["#C44E52" if v > 0 else "#4C72B0" for v in top.values]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top.index, top.values, color=colors)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_title("SHAP local — facteurs de risque d'un client\n(rouge = augmente le churn, bleu = réduit)")
    ax.set_xlabel("Contribution SHAP à la proba de churn")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    return top[::-1]


def main():
    model = load_model()
    X_train, X_test, y_train, y_test = load_split()
    F = config.FIGURES_DIR

    print("== Permutation importance ==", flush=True)
    perm = permutation_fig(model, X_test, y_test, F / "20_permutation_importance.png")
    print(perm.head(10).round(4).to_string())

    print("\n== SHAP global ==", flush=True)
    Xs = X_test.sample(min(200, len(X_test)), random_state=config.RANDOM_STATE)
    explainer, names = shap_global_fig(model, Xs, F / "21_shap_summary.png")

    print("== SHAP local (client a haut risque) ==", flush=True)
    proba = model.predict_proba(X_test)[:, 1]
    idx = X_test.index[np.argmax(proba)]
    print(f"Client {idx} | proba churn = {proba.max():.2f}")
    top = shap_local_fig(model, explainer, X_test.loc[[idx]], names, F / "22_shap_local.png")
    print(top.round(4).to_string())
    print("\nFigures -> reports/figures/20..22")


if __name__ == "__main__":
    main()
