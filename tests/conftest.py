"""Fixtures compartilhadas pelos testes."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_prep import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from src.train import build_candidates, split_data


@pytest.fixture(scope="session")
def dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Dataset pequeno e deterministico, com sinal claro (contrato mensal => churn)."""
    rng = np.random.default_rng(0)
    n = 400
    tenure = rng.integers(0, 72, n)
    monthly = rng.normal(65, 20, n).clip(15, 150)
    X = pd.DataFrame(
        {
            "tenure_months": tenure,
            "monthly_charges": monthly,
            "total_charges": monthly * np.maximum(tenure, 1),
            "num_support_calls": rng.poisson(1.5, n),
            "contract_type": rng.choice(["month-to-month", "one_year", "two_year"], n),
            "internet_service": rng.choice(["fiber", "dsl", "none"], n),
            "tech_support": rng.choice(["yes", "no"], n),
            "payment_method": rng.choice(["credit_card", "bank_transfer", "electronic_check"], n),
            "has_dependents": rng.choice(["yes", "no"], n),
            "senior_citizen": rng.choice([0, 1], n),
        }
    )[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    logit = -1.0 + 2.0 * (X["contract_type"] == "month-to-month") - 0.03 * X["tenure_months"]
    y = pd.Series(rng.binomial(1, 1 / (1 + np.exp(-logit))), name="churn")
    return X, y


@pytest.fixture(scope="session")
def fitted_logreg(dataset):
    X, y = dataset
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = build_candidates()["logistic_regression"].fit(X_train, y_train)
    return model, X_train, X_test, y_train, y_test
