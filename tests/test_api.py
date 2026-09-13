"""Testes da API definida em api/main.py."""

from __future__ import annotations

import os

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def make_customer(**overrides) -> dict:
    base = {
        "tenure_months": 24,
        "monthly_charges": 60.0,
        "total_charges": 1440.0,
        "num_support_calls": 1,
        "contract_type": "one_year",
        "internet_service": "dsl",
        "tech_support": "yes",
        "payment_method": "credit_card",
        "has_dependents": "yes",
        "senior_citizen": 0,
    }
    base.update(overrides)
    return base


def test_predict_returns_valid_shape():
    response = client.post("/predict", json=make_customer())
    assert response.status_code == 200
    data = response.json()
    assert data["churn_prediction"] in (0, 1)
    assert 0.0 <= data["churn_probability"] <= 1.0


def test_predict_high_risk_customer_scores_higher_than_low_risk():
    high_risk = make_customer(
        tenure_months=1,
        contract_type="month-to-month",
        tech_support="no",
        payment_method="electronic_check",
        num_support_calls=5,
        senior_citizen=1,
        has_dependents="no",
    )
    low_risk = make_customer(
        tenure_months=60,
        contract_type="two_year",
        tech_support="yes",
        payment_method="credit_card",
        num_support_calls=0,
        senior_citizen=0,
        has_dependents="yes",
    )

    high_prob = client.post("/predict", json=high_risk).json()["churn_probability"]
    low_prob = client.post("/predict", json=low_risk).json()["churn_probability"]

    assert high_prob > low_prob


def test_generate_report_creates_scored_csv():
    response = client.post("/generate-report")
    assert response.status_code == 200
    data = response.json()
    assert data["rows"] > 0
    assert os.path.exists(data["output_path"])
