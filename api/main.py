"""API REST d'inférence churn (FastAPI).

Service Front/API/Modèle : charge le pipeline sérialisé (best_model.pkl) et expose :
- GET  /health      : état du service + modèle chargé
- GET  /model-info  : métadonnées du modèle (nom, seuil, métriques, features)
- POST /predict     : reçoit les features d'un client -> proba + décision de churn

Lancement : uvicorn api.main:app --reload   (depuis la racine du repo)
"""
import json
import pickle
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src import config

MODEL = None
META = {}
LOAD_ERROR = None
try:
    with open(config.MODELS_DIR / "best_model.pkl", "rb") as f:
        MODEL = pickle.load(f)
    with open(config.MODELS_DIR / "model_meta.json") as f:
        META = json.load(f)
except Exception as e:
    LOAD_ERROR = str(e)

THRESHOLD = float(META.get("threshold", 0.5))

app = FastAPI(
    title="Churn Prediction API",
    description="Service d'inference pour la prediction de resiliation client (churn).",
    version="1.0.0",
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
        "numeric_features": config.NUMERIC_FEATURES,
        "categorical_features": config.CATEGORICAL_FEATURES,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(features: CustomerFeatures):
    if MODEL is None:
        raise HTTPException(status_code=503, detail=f"Modele non charge : {LOAD_ERROR}")
    try:
        row = features.model_dump()
        X = pd.DataFrame([row])[config.ALL_FEATURES]
        proba = float(MODEL.predict_proba(X)[0, 1])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur de prediction : {e}")
    pred = int(proba >= THRESHOLD)
    return PredictionResponse(
        churn_probability=round(proba, 4),
        churn_prediction=pred,
        label="Churn probable" if pred else "Client stable",
        risk_level=_risk_level(proba),
        threshold=round(THRESHOLD, 4),
        model_name=META.get("model_name", "unknown"),
    )
