"""Tests de l'API churn (independants du dashboard).

Usage : python -m api.test_api   (ou: pytest api/test_api.py)
"""
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)
EXAMPLE = app.openapi()["components"]["schemas"]["CustomerFeatures"]["example"]


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is True


def test_model_info():
    r = client.get("/model-info")
    assert r.status_code == 200
    assert "metrics_test" in r.json()


def test_model_info_top_drivers():
    r = client.get("/model-info")
    assert r.status_code == 200
    assert len(r.json()["top_drivers"]) >= 5


def test_predict_valid():
    r = client.post("/predict", json=EXAMPLE)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["churn_prediction"] in (0, 1)


def test_predict_discriminates():
    risky = client.post("/predict", json=EXAMPLE).json()["churn_probability"]
    good = dict(EXAMPLE)
    good.update({"tenure_months": 48, "csat_score": 5.0, "payment_failures": 0,
                 "monthly_logins": 25, "usage_growth_rate": 0.2, "nps_score": 60,
                 "last_login_days_ago": 1, "survey_response": "Satisfied"})
    stable = client.post("/predict", json=good).json()["churn_probability"]
    assert risky > stable


def test_predict_missing_field():
    bad = dict(EXAMPLE); bad.pop("csat_score")
    assert client.post("/predict", json=bad).status_code == 422


def test_predict_wrong_type():
    bad = dict(EXAMPLE); bad["csat_score"] = "abc"
    assert client.post("/predict", json=bad).status_code == 422


def test_predict_batch():
    r = client.post("/predict-batch", json={"customers": [EXAMPLE, EXAMPLE]})
    assert r.status_code == 200
    assert r.json()["n"] == 2
    assert all(0 <= x["churn_probability"] <= 1 for x in r.json()["results"])


def test_predict_batch_empty():
    assert client.post("/predict-batch", json={"customers": []}).status_code == 400


def test_explain():
    r = client.post("/explain", json=EXAMPLE)
    assert r.status_code == 200
    assert len(r.json()["factors"]) >= 1
    assert r.json()["factors"][0]["direction"] in ("augmente", "reduit")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in tests:
        fn(); print(f"PASS {fn.__name__}")
    print(f"\n{len(tests)} tests API OK.")
