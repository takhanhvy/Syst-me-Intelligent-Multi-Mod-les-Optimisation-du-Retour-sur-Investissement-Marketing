"""Préparation des données : ColumnTransformer anti-leakage + split train/test.

Le preprocessing est encapsulé dans un sklearn Pipeline avec le modèle, donc il est
*fit* uniquement sur le train à chaque fold/entraînement → aucune fuite de données.
"""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config


def build_preprocessor():
    """ColumnTransformer : numériques (impute médiane + scale), catégorielles (impute 'None' + OneHot)."""
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="None")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric, config.NUMERIC_FEATURES),
        ("cat", categorical, config.CATEGORICAL_FEATURES),
    ])


def get_feature_names(preprocessor):
    """Noms des colonnes après transformation (pour l'interprétabilité)."""
    names = list(config.NUMERIC_FEATURES)
    ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    names += list(ohe.get_feature_names_out(config.CATEGORICAL_FEATURES))
    return names


def load_split():
    """Charge le dataset et renvoie X_train, X_test, y_train, y_test (split stratifié)."""
    df = config.load_raw()
    X = df[config.ALL_FEATURES].copy()
    y = df[config.TARGET].copy()
    return train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        stratify=y,
        random_state=config.RANDOM_STATE,
    )
