"""API REST d'inference churn (FastAPI).

Service Front/API/Modele : charge le pipeline serialise (best_model.pkl) et expose :
- GET  /health        : etat du service + modele charge
- GET  /model-info    : metadonnees (nom, seuil, metriques, top facteurs de churn)
- POST /predict       : un client -> proba + decision de churn
- POST /predict-batch : N clients -> liste de predictions (KPI dashboard)
- POST /explain       : un client -> top facteurs SHAP (le 'pourquoi')

Lancement : uvicorn api.main:app --reload   (depuis la racine du repo)
"""
import json
import pickle
from typing import List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src import config
from src.data_preprocessing import get_feature_names

MODEL = None
META = {}
LOAD_ERROR = None
TOP_DRIVERS = []
_EXPLAINER = None

try:
    with open(config.MODELS_DIR / "best_model.pkl", "rb") as f:
        MODEL = pickle.load(f)
    with open(config.MODELS_DIR / "model_meta.json") as f:
        META = json.load(f)
    # Top facteurs de churn (importance native du RF agregee aux variables d'origine)
    prep = MODEL.named_steps["prep"]
    rf = MODEL.named_steps["model"]
    names = get_feature_names(prep)
    agg = {}
    for nm, imp in zip(names, rf.feature_importances_):
        if nm in config.NUMERIC_FEATURES:
            orig = nm
        else:
            orig = next((c for c in config.CATEGORICAL_FEATURES if nm.startswith(c + "_")), nm)
        agg[orig] = agg.get(orig, 0.0) + float(imp)
    total = sum(agg.values()) or 1.0
    TOP_DRIVERS = [
        {"feature": k, "importance": round(v / total, 4)}
        for k, v in sorted(agg.items(), key=lambda x: x[1], reverse=True)[:8]
    ]
except Exception as e:
    LOAD_ERROR = str(e)

THRESHOLD = float(META.get("threshold", 0.5))

app = FastAPI(
    title="Churn Prediction API",
    description="Service d'inference pour la prediction de resiliation client (churn).",
    version="1.1.0",
)


class CustomerFeatures(BaseModel):
    age: float
    tenure_months: float
    monthly_logins: float
    weekly_active_days: float
    avg_session_time: float
    features_used: float
    usage_growth_rate: float
    last_login_days_ago: float
    monthly_fee: float
    total_revenue: float
    payment_failures: float
    support_tickets: float
    avg_resolution_time: float
    csat_score: float
    escalations: float
    email_open_rate: float
    marketing_click_rate: float
    nps_score: float
    referral_count: float
    gender: str
    country: str
    city: str
    customer_segment: str
    signup_channel: str
    contract_type: str
    payment_method: str
    discount_applied: str
    price_increase_last_3m: str
    complaint_type: Optional[str] = Field(default=None, description="None si pas de plainte")
    survey_response: str

    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 45, "tenure_months": 4, "monthly_logins": 3,
                "weekly_active_days": 2, "avg_session_time": 8.5, "features_used": 2,
                "usage_growth_rate": -0.3, "last_login_days_ago": 25, "monthly_fee": 60,
                "total_revenue": 240, "payment_failures": 3, "support_tickets": 5,
                "avg_resolution_time": 30.0, "csat_score": 2.0, "escalations": 2,
                "email_open_rate": 0.2, "marketing_click_rate": 0.05, "nps_score": -10,
                "referral_count": 0, "gender": "Female", "country": "Germany",
                "city": "New York", "customer_segment": "SME", "signup_channel": "Referral",
                "contract_type": "Monthly", "payment_method": "Card",
                "discount_applied": "No", "price_increase_last_3m": "Yes",
                "complaint_type": "Billing", "survey_response": "Unsatisfied",
            }
        }
    }


class BatchRequest(BaseModel):
    customers: List[CustomerFeatures]


class PredictionResponse(BaseModel):
    churn_probability: float
    churn_prediction: int
    label: str
    risk_level: str
    threshold: float
    model_name: str


def _risk_level(p: float) -> str:
    if p >= THRESHOLD:
        return "Eleve"
    if p >= THRESHOLD * 0.6:
        return "Moyen"
    return "Faible"


def _to_df(items):
    return pd.DataFrame([c.model_dump() for c in items])[config.ALL_FEATURES]


@app.get("/health")
def health():
    return {
        "status": "ok" if MODEL is not None else "error",
        "model_loaded": MODEL is not None,
        "model_name": META.get("model_name"),
        "error": LOAD_ERROR,
    }


@app.get("/model-info")
def model_info():
    if MODEL is None:
        raise HTTPException(status_code=503, detail=f"Modele non charge : {LOAD_ERROR}")
    return {
        "model_name": META.get("model_name"),
        "threshold": THRESHOLD,
        "metrics_test": META.get("metrics_test"),
        "n_features": len(config.ALL_FEATURES),
        "top_drivers": TOP_DRIVERS,
        "numeric_features": config.NUMERIC_FEATURES,
        "categorical_features": config.CATEGORICAL_FEATURES,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(features: CustomerFeatures):
    if MODEL is None:
        raise HTTPException(status_code=503, detail=f"Modele non charge : {LOAD_ERROR}")
    try:
        X = _to_df([features])
        proba = float(MODEL.predict_proba(X)[0, 1])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur de prediction : {e}")
    pred = int(proba >= THRESHOLD)
    return PredictionResponse(
        churn_probability=round(proba, 4), churn_prediction=pred,
        label="Churn probable" if pred else "Client stable",
        risk_level=_risk_level(proba), threshold=round(THRESHOLD, 4),
        model_name=META.get("model_name", "unknown"),
    )


@app.post("/predict-batch")
def predict_batch(req: BatchRequest):
    if MODEL is None:
        raise HTTPException(status_code=503, detail=f"Modele non charge : {LOAD_ERROR}")
    if not req.customers:
        raise HTTPException(status_code=400, detail="Liste de clients vide.")
    try:
        X = _to_df(req.customers)
        proba = MODEL.predict_proba(X)[:, 1]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur de prediction batch : {e}")
    results = [
        {"churn_probability": round(float(p), 4),
         "churn_prediction": int(p >= THRESHOLD),
         "risk_level": _risk_level(float(p))}
        for p in proba
    ]
    return {"n": len(results), "threshold": round(THRESHOLD, 4), "results": results}


@app.post("/explain")
def explain(features: CustomerFeatures, top: int = 8):
    """Top facteurs SHAP locaux pour un client (le 'pourquoi')."""
    global _EXPLAINER
    if MODEL is None:
        raise HTTPException(status_code=503, detail=f"Modele non charge : {LOAD_ERROR}")
    try:
        import shap
        prep = MODEL.named_steps["prep"]
        rf = MODEL.named_steps["model"]
        Xt = prep.transform(_to_df([features]))
        if hasattr(Xt, "toarray"):
            Xt = Xt.toarray()
        feat_names = get_feature_names(prep)
        if _EXPLAINER is None:
            _EXPLAINER = shap.TreeExplainer(rf)
        sv = _EXPLAINER.shap_values(Xt, check_additivity=False)
        sv1 = sv[1] if isinstance(sv, list) else (sv[:, :, 1] if getattr(sv, "ndim", 2) == 3 else sv)
        contrib = np.asarray(sv1)[0]
        order = np.argsort(np.abs(contrib))[::-1][:top]
        factors = [
            {"feature": feat_names[i], "contribution": round(float(contrib[i]), 4),
             "direction": "augmente" if contrib[i] > 0 else "reduit"}
            for i in order
        ]
        proba = float(MODEL.predict_proba(_to_df([features]))[0, 1])
        return {"churn_probability": round(proba, 4), "factors": factors}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur d'explication : {e}")
