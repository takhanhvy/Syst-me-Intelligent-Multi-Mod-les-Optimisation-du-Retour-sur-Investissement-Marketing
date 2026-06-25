"""Définition des 4 modèles, chacun encapsulé avec le préprocessing dans un Pipeline.

Gestion du déséquilibre (churn ~10 %) :
- LogisticRegression / RandomForest : class_weight="balanced"
- XGBoost : scale_pos_weight = n_neg / n_pos
- MLP : pas de class_weight natif → compensé par l'ajustement du seuil (cf. evaluation.py)
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

from . import config
from .data_preprocessing import build_preprocessor


def _pipe(model):
    return Pipeline([("prep", build_preprocessor()), ("model", model)])


def get_models(y_train):
    """Renvoie un dict {nom: pipeline} des 4 modèles à comparer."""
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    spw = n_neg / max(n_pos, 1)
    rs = config.RANDOM_STATE

    models = {
        "LogisticRegression": _pipe(LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=rs,
        )),
        "RandomForest": _pipe(RandomForestClassifier(
            n_estimators=200, max_depth=None, min_samples_leaf=2,
            class_weight="balanced", n_jobs=-1, random_state=rs,
        )),
        "XGBoost": _pipe(_make_xgb(spw, rs)),
        "MLP": _pipe(MLPClassifier(
            hidden_layer_sizes=(64, 32), activation="relu", alpha=1e-3,
            early_stopping=True, n_iter_no_change=10, max_iter=300, random_state=rs,
        )),
    }
    return models


def _make_xgb(scale_pos_weight, random_state):
    from xgboost import XGBClassifier
    return XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss", n_jobs=-1, random_state=random_state,
    )
