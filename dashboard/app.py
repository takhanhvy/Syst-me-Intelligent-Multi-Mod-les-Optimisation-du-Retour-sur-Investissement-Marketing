"""Dashboard decisionnel churn (Streamlit) - oriente utilisateur metier.

Outil pour responsable marketing / CRM / direction financiere :
- KPI (clients a risque, revenu global a risque)
- Priorisation des clients a contacter
- Simulateur temps reel + pourquoi (facteurs SHAP) via l'API

Le dashboard appelle l'API REST (Front / API / Modele). Aucun visuel scientifique ici.
Lancement : streamlit run dashboard/app.py   (API lancee en parallele)
"""
import os
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Pilotage Retention Client", page_icon="📊", layout="wide")

# Actions metier suggerees selon le facteur de risque
ACTIONS = {
    "csat_score": "Programmer un appel satisfaction",
    "nps_score": "Geste commercial / ecoute client",
    "payment_failures": "Verifier le moyen de paiement, relance douce",
    "tenure_months": "Renforcer l'onboarding (client recent)",
    "monthly_logins": "Campagne de re-engagement",
    "last_login_days_ago": "Relance : client inactif",
    "usage_growth_rate": "Proposer une demo des fonctionnalites",
    "support_tickets": "Prioriser la resolution des tickets",
    "avg_resolution_time": "Accelerer le support",
    "weekly_active_days": "Stimuler l'usage hebdomadaire",
}


def api_get(path):
    return requests.get(f"{API_URL}{path}", timeout=30)


def api_post(path, payload):
    return requests.post(f"{API_URL}{path}", json=payload, timeout=120)


@st.cache_data(show_spinner=False)
def load_data():
    df = config.load_raw()
    return df


def to_payload(df):
    sub = df[config.ALL_FEATURES].copy()
    sub = sub.astype(object).where(pd.notna(sub), None)  # NaN -> None (JSON)
    return sub.to_dict(orient="records")


@st.cache_data(show_spinner="Scoring des clients via l'API...")
def score_all(_df, api_url):
    payload = {"customers": to_payload(_df)}
    r = requests.post(f"{api_url}/predict-batch", json=payload, timeout=300)
    r.raise_for_status()
    res = r.json()["results"]
    out = _df.copy()
    out["churn_proba"] = [x["churn_probability"] for x in res]
    out["churn_pred"] = [x["churn_prediction"] for x in res]
    out["risk_level"] = [x["risk_level"] for x in res]
    out["revenue_at_risk"] = out["churn_proba"] * out["total_revenue"]
    return out


def check_api():
    try:
        r = api_get("/health")
        return r.status_code == 200 and r.json().get("model_loaded")
    except Exception:
        return False


# ---------------- Sidebar ----------------
st.sidebar.title("📊 Pilotage Rétention")
page = st.sidebar.radio("Navigation", [
    "Vue d'ensemble", "Clients à risque", "Simulateur client", "Confiance du modèle",
])
st.sidebar.caption(f"API : {API_URL}")

if not check_api():
    st.error(f"⚠️ L'API n'est pas joignable sur {API_URL}.\n\n"
             f"Lancez-la d'abord :  `uvicorn api.main:app`")
    st.stop()

df = load_data()
scored = score_all(df, API_URL)
THR = api_get("/model-info").json()["threshold"]


# ---------------- Page 1 : Vue d'ensemble ----------------
if page == "Vue d'ensemble":
    st.title("Vue d'ensemble — risque de résiliation")
    n = len(scored)
    n_risk = int((scored["churn_pred"] == 1).sum())
    rev_risk = float(scored["revenue_at_risk"].sum())
    exp_churn = float(scored["churn_proba"].sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Clients", f"{n:,}".replace(",", " "))
    c2.metric("Clients à risque", f"{n_risk:,}".replace(",", " "), f"{n_risk/n:.1%}")
    c3.metric("Churns attendus", f"{exp_churn:,.0f}".replace(",", " "))
    c4.metric("Revenu global à risque", f"{rev_risk:,.0f} €".replace(",", " "))

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Répartition par niveau de risque")
        order = ["Faible", "Moyen", "Eleve"]
        d = scored["risk_level"].value_counts().reindex(order).fillna(0).reset_index()
        d.columns = ["Niveau", "Clients"]
        fig = px.bar(d, x="Niveau", y="Clients", color="Niveau",
                     color_discrete_map={"Faible": "#4C72B0", "Moyen": "#DD8452", "Eleve": "#C44E52"})
        fig.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig, width="stretch")
    with col2:
        st.subheader("Revenu à risque par segment")
        seg = scored.groupby("customer_segment")["revenue_at_risk"].sum().sort_values(ascending=False).reset_index()
        fig = px.bar(seg, x="customer_segment", y="revenue_at_risk",
                     labels={"customer_segment": "Segment", "revenue_at_risk": "Revenu à risque (€)"})
        fig.update_layout(height=350)
        st.plotly_chart(fig, width="stretch")

    st.info(f"Le revenu à risque correspond à la somme, sur tous les clients, de "
            f"**probabilité de churn × revenu du client**. Seuil de décision : {THR:.2f}.")


# ---------------- Page 2 : Clients à risque ----------------
elif page == "Clients à risque":
    st.title("Clients à risque — priorisation des actions")
    st.caption("Triés par revenu à risque décroissant : qui contacter en priorité.")

    colf1, colf2, colf3 = st.columns(3)
    levels = colf1.multiselect("Niveau de risque", ["Eleve", "Moyen", "Faible"], default=["Eleve", "Moyen"])
    segs = colf2.multiselect("Segment", sorted(scored["customer_segment"].unique()),
                             default=sorted(scored["customer_segment"].unique()))
    topn = colf3.slider("Nombre de clients affichés", 10, 500, 50, step=10)

    view = scored[scored["risk_level"].isin(levels) & scored["customer_segment"].isin(segs)]
    view = view.sort_values("revenue_at_risk", ascending=False).head(topn)

    show = view[["customer_id", "customer_segment", "contract_type", "tenure_months",
                 "csat_score", "payment_failures", "total_revenue",
                 "churn_proba", "risk_level", "revenue_at_risk"]].copy()
    show["churn_proba"] = (show["churn_proba"] * 100).round(1)
    show["revenue_at_risk"] = show["revenue_at_risk"].round(0)
    show = show.rename(columns={
        "customer_id": "Client", "customer_segment": "Segment", "contract_type": "Contrat",
        "tenure_months": "Ancienneté", "csat_score": "CSAT", "payment_failures": "Échecs paiement",
        "total_revenue": "Revenu (€)", "churn_proba": "Proba churn (%)",
        "risk_level": "Risque", "revenue_at_risk": "Revenu à risque (€)"})
    st.dataframe(show, width="stretch", height=480, hide_index=True)

    st.metric("Revenu à risque (sélection)", f"{view['revenue_at_risk'].sum():,.0f} €".replace(",", " "))
    st.download_button("⬇️ Exporter la liste (CSV)", show.to_csv(index=False).encode("utf-8"),
                       "clients_a_risque.csv", "text/csv")


# ---------------- Page 3 : Simulateur ----------------
elif page == "Simulateur client":
    st.title("Simulateur — prédiction en temps réel")
    st.caption("Saisissez (ou modifiez) un profil client et obtenez la probabilité de churn + les facteurs.")

    base = df.sample(1, random_state=7).iloc[0]
    with st.form("client"):
        c1, c2, c3 = st.columns(3)
        vals = {}
        with c1:
            vals["tenure_months"] = st.number_input("Ancienneté (mois)", 0, 120, int(base["tenure_months"]))
            vals["monthly_logins"] = st.number_input("Connexions/mois", 0, 100, int(base["monthly_logins"]))
            vals["weekly_active_days"] = st.number_input("Jours actifs/sem.", 0, 7, int(base["weekly_active_days"]))
            vals["last_login_days_ago"] = st.number_input("Dernière connexion (j)", 0, 365, int(base["last_login_days_ago"]))
            vals["usage_growth_rate"] = st.slider("Croissance d'usage", -1.0, 1.0, float(base["usage_growth_rate"]))
            vals["features_used"] = st.number_input("Fonctionnalités utilisées", 0, 50, int(base["features_used"]))
            vals["avg_session_time"] = st.number_input("Durée session (min)", 0.0, 120.0, float(base["avg_session_time"]))
        with c2:
            vals["csat_score"] = st.slider("CSAT (1-5)", 1.0, 5.0, float(base["csat_score"]))
            vals["nps_score"] = st.number_input("NPS (-100 à 100)", -100, 100, int(base["nps_score"]))
            vals["payment_failures"] = st.number_input("Échecs de paiement", 0, 20, int(base["payment_failures"]))
            vals["support_tickets"] = st.number_input("Tickets support", 0, 50, int(base["support_tickets"]))
            vals["avg_resolution_time"] = st.number_input("Tps résolution (h)", 0.0, 200.0, float(base["avg_resolution_time"]))
            vals["escalations"] = st.number_input("Escalades", 0, 20, int(base["escalations"]))
            vals["email_open_rate"] = st.slider("Taux ouverture email", 0.0, 1.0, float(base["email_open_rate"]))
            vals["marketing_click_rate"] = st.slider("Taux clic marketing", 0.0, 1.0, float(base["marketing_click_rate"]))
        with c3:
            vals["monthly_fee"] = st.number_input("Abonnement (€)", 0, 1000, int(base["monthly_fee"]))
            vals["total_revenue"] = st.number_input("Revenu total (€)", 0, 100000, int(base["total_revenue"]))
            vals["referral_count"] = st.number_input("Parrainages", 0, 50, int(base["referral_count"]))
            vals["age"] = st.number_input("Âge", 18, 100, int(base["age"]))
            vals["gender"] = st.selectbox("Genre", sorted(df["gender"].unique()))
            vals["country"] = st.selectbox("Pays", sorted(df["country"].unique()))
            vals["city"] = st.selectbox("Ville", sorted(df["city"].unique()))
            vals["customer_segment"] = st.selectbox("Segment", sorted(df["customer_segment"].unique()))
            vals["signup_channel"] = st.selectbox("Canal d'acquisition", sorted(df["signup_channel"].unique()))
            vals["contract_type"] = st.selectbox("Type de contrat", sorted(df["contract_type"].unique()))
            vals["payment_method"] = st.selectbox("Moyen de paiement", sorted(df["payment_method"].unique()))
            vals["discount_applied"] = st.selectbox("Remise appliquée", sorted(df["discount_applied"].unique()))
            vals["price_increase_last_3m"] = st.selectbox("Hausse prix (3 mois)", sorted(df["price_increase_last_3m"].unique()))
            comp = sorted([str(x) for x in df["complaint_type"].dropna().unique()])
            vals["complaint_type"] = st.selectbox("Type de plainte", ["(aucune)"] + comp)
            vals["survey_response"] = st.selectbox("Réponse enquête", sorted(df["survey_response"].unique()))
        submitted = st.form_submit_button("🔮 Prédire le risque", width="stretch")

    if submitted:
        if vals["complaint_type"] == "(aucune)":
            vals["complaint_type"] = None
        r = api_post("/predict", vals)
        if r.status_code != 200:
            st.error(f"Erreur API : {r.text}")
        else:
            res = r.json()
            proba = res["churn_probability"]
            colg, cole = st.columns([1, 1.3])
            with colg:
                gauge = go.Figure(go.Indicator(
                    mode="gauge+number", value=proba * 100,
                    number={"suffix": " %"},
                    title={"text": f"Probabilité de churn — {res['risk_level']}"},
                    gauge={"axis": {"range": [0, 100]},
                           "bar": {"color": "#C44E52" if res["churn_prediction"] else "#4C72B0"},
                           "threshold": {"line": {"color": "black", "width": 3},
                                         "value": THR * 100}}))
                gauge.update_layout(height=320)
                st.plotly_chart(gauge, width="stretch")
                st.metric("Décision", res["label"])
            with cole:
                st.subheader("Pourquoi ? — facteurs de risque")
                ex = api_post("/explain", vals)
                if ex.status_code == 200:
                    for fct in ex.json()["factors"][:6]:
                        f = fct["feature"]
                        up = fct["direction"] == "augmente"
                        icon = "🔴" if up else "🟢"
                        base_feat = f.split("_")[0] if f not in ACTIONS else f
                        action = ACTIONS.get(f, "")
                        txt = f"{icon} **{f}** — {'augmente' if up else 'réduit'} le risque"
                        if up and action:
                            txt += f"  → _{action}_"
                        st.write(txt)
                else:
                    st.caption("Explication indisponible.")


# ---------------- Page 4 : Confiance du modèle ----------------
elif page == "Confiance du modèle":
    st.title("Confiance du modèle")
    info = api_get("/model-info").json()
    m = info["metrics_test"]
    st.write(f"Modèle en production : **{info['model_name']}**  ·  seuil de décision **{info['threshold']:.2f}**")

    c1, c2, c3 = st.columns(3)
    c1.metric("Churners détectés (Recall)", f"{m['Recall']:.0%}",
              help="Part des clients qui résilient que le modèle réussit à repérer.")
    c2.metric("Justesse des alertes (Precision)", f"{m['Precision']:.0%}",
              help="Part des clients alertés qui résilient réellement.")
    c3.metric("Pouvoir de discrimination (ROC-AUC)", f"{m['ROC_AUC']:.2f}",
              help="0.5 = hasard, 1.0 = parfait.")

    st.divider()
    st.subheader("Facteurs de churn les plus influents")
    st.caption("Ce qui pèse le plus dans la décision du modèle, à l'échelle de la base clients.")
    drv = pd.DataFrame(info["top_drivers"])
    fig = px.bar(drv.sort_values("importance"), x="importance", y="feature", orientation="h",
                 labels={"importance": "Poids relatif", "feature": ""})
    fig.update_layout(height=400)
    st.plotly_chart(fig, width="stretch")
    st.info("Lecture métier : un **CSAT bas**, une **faible ancienneté**, **peu de connexions** "
            "et des **échecs de paiement** sont les signaux de départ les plus forts.")
