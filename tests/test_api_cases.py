"""Casos adicionais da API: perfis de cliente, validacao de entrada e limites."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.main import app
from src.insights import PERSONAS

client = TestClient(app)


def post_predict(payload: dict):
    return client.post("/predict", json=payload)


# ------------------------------------------------------ perfis de cliente
@pytest.mark.parametrize("name", ["Recém-chegado insatisfeito", "Muitas ligações ao suporte"])
def test_high_risk_personas_are_predicted_as_churn(name):
    data = post_predict(PERSONAS[name]).json()
    assert data["churn_prediction"] == 1
    assert data["churn_probability"] > 0.5


@pytest.mark.parametrize("name", ["Cliente fiel", "Família estável", "Contrato anual médio",
                                  "Plano básico sem internet"])
def test_low_risk_personas_are_predicted_as_retained(name):
    data = post_predict(PERSONAS[name]).json()
    assert data["churn_prediction"] == 0
    assert data["churn_probability"] < 0.5


@pytest.mark.parametrize("name", list(PERSONAS))
def test_prediction_is_consistent_with_probability(name):
    data = post_predict(PERSONAS[name]).json()
    assert data["churn_prediction"] == int(data["churn_probability"] >= 0.5)


def test_same_input_gives_same_output():
    payload = PERSONAS["Mês a mês veterano"]
    assert post_predict(payload).json() == post_predict(payload).json()


# --------------------------------------------- efeito de cada fator isolado
BASE = PERSONAS["Contrato anual médio"]


def prob(**overrides) -> float:
    return post_predict({**BASE, **overrides}).json()["churn_probability"]


def test_month_to_month_raises_risk_vs_two_year():
    assert prob(contract_type="month-to-month") > prob(contract_type="two_year")


def test_more_support_calls_raise_risk():
    assert prob(num_support_calls=6) > prob(num_support_calls=0)


def test_longer_tenure_lowers_risk():
    assert prob(tenure_months=1, total_charges=65.0) > prob(tenure_months=70, total_charges=4550.0)


def test_tech_support_lowers_risk():
    assert prob(tech_support="no") > prob(tech_support="yes")


def test_electronic_check_raises_risk_vs_credit_card():
    assert prob(payment_method="electronic_check") > prob(payment_method="credit_card")


# -------------------------------------------------------- entrada invalida
@pytest.mark.parametrize("missing", list(BASE))
def test_missing_field_returns_422(missing):
    payload = {k: v for k, v in BASE.items() if k != missing}
    assert post_predict(payload).status_code == 422


@pytest.mark.parametrize(
    "field, value",
    [("tenure_months", "muito"), ("monthly_charges", "caro"), ("num_support_calls", "varias"),
     ("senior_citizen", "talvez"), ("tenure_months", 3.7), ("tenure_months", None)],
)
def test_wrong_type_returns_422(field, value):
    assert post_predict({**BASE, field: value}).status_code == 422


def test_empty_body_returns_422():
    assert post_predict({}).status_code == 422


def test_extra_fields_are_ignored():
    response = post_predict({**BASE, "campo_inventado": "x"})
    assert response.status_code == 200


# --------------------------------------------------------------- limites
@pytest.mark.parametrize(
    "overrides",
    [
        dict(tenure_months=0, total_charges=0.0),
        dict(tenure_months=71, total_charges=10000.0),
        dict(monthly_charges=15.0),
        dict(monthly_charges=150.0),
        dict(num_support_calls=0),
        dict(num_support_calls=15),
        dict(senior_citizen=1),
    ],
)
def test_boundary_values_still_return_valid_probability(overrides):
    response = post_predict({**BASE, **overrides})
    assert response.status_code == 200
    assert 0.0 <= response.json()["churn_probability"] <= 1.0


def test_unknown_category_does_not_crash_the_model():
    """OneHotEncoder(handle_unknown='ignore'): categoria nova vira linha de zeros.

    Documenta o comportamento atual: a API NAO rejeita categorias fora do esperado.
    """
    response = post_predict({**BASE, "contract_type": "contrato_novo"})
    assert response.status_code == 200


# --------------------------------------------------------------- rotas
def test_unknown_route_returns_404():
    assert client.get("/nao-existe").status_code == 404


def test_predict_does_not_accept_get():
    assert client.get("/predict").status_code == 405


def test_health_does_not_accept_post():
    assert client.post("/health").status_code == 405


def test_openapi_lists_all_endpoints():
    paths = client.get("/openapi.json").json()["paths"]
    assert {"/health", "/predict", "/generate-report"} <= set(paths)
