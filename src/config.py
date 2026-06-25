"""Configuration centrale du projet : chemins, constantes, listes de features."""
from pathlib import Path

# --- Chemins ---
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATA_FILE = DATA_DIR / "customer_churn_business_dataset.csv"
MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "reports" / "figures"

for _d in (MODELS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- Reproductibilité ---
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

# --- Cible & colonnes à exclure ---
TARGET = "churn"
ID_COLS = ["customer_id"]

# --- Features ---
# Catégorielles -> OneHotEncoder
CATEGORICAL_FEATURES = [
    "gender",
    "country",
    "city",
    "customer_segment",
    "signup_channel",
    "contract_type",
    "payment_method",
    "discount_applied",
    "price_increase_last_3m",
    "complaint_type",      # contient des NaN -> imputés "None"
    "survey_response",
]

# Numériques -> imputation médiane + StandardScaler
NUMERIC_FEATURES = [
    "age",
    "tenure_months",
    "monthly_logins",
    "weekly_active_days",
    "avg_session_time",
    "features_used",
    "usage_growth_rate",
    "last_login_days_ago",
    "monthly_fee",
    "total_revenue",
    "payment_failures",
    "support_tickets",
    "avg_resolution_time",
    "csat_score",
    "escalations",
    "email_open_rate",
    "marketing_click_rate",
    "nps_score",
    "referral_count",
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_raw():
    """Charge le dataset brut."""
    import pandas as pd
    return pd.read_csv(DATA_FILE)
