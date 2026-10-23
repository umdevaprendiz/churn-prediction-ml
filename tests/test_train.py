"""Testes do treino/comparacao de modelos definidos em src/train.py."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_prep import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from src.train import build_candidates, split_data


def make_dataset(n: int = 200) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {
            "tenure_months": rng.integers(0, 72, n),
            "monthly_charges": rng.normal(65, 20, n).clip(15, 150),
            "total_charges": rng.normal(2000, 800, n).clip(10),
            "num_support_calls": rng.poisson(1.5, n),
            "contract_type": rng.choice(["month-to-month", "one_year", "two_year"], n),
            "internet_service": rng.choice(["fiber", "dsl", "none"], n),
            "tech_support": rng.choice(["yes", "no"], n),
            "payment_method": rng.choice(["credit_card", "bank_transfer", "electronic_check"], n),
            "has_dependents": rng.choice(["yes", "no"], n),
            "senior_citizen": rng.choice([0, 1], n),
        }
    )[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = pd.Series((X["contract_type"] == "month-to-month").astype(int) ^ rng.binomial(1, 0.1, n), name="churn")
    return X, y


def test_build_candidates_has_three_models():
    assert set(build_candidates()) == {"logistic_regression", "random_forest", "gradient_boosting"}


def test_candidates_do_not_share_preprocessor():
    candidates = build_candidates()
    preprocessors = [p.named_steps["preprocessor"] for p in candidates.values()]
    assert len({id(p) for p in preprocessors}) == len(preprocessors)


def test_split_data_is_stratified():
    X, y = make_dataset()
    _, _, y_train, y_test = split_data(X, y)
    assert abs(y_train.mean() - y_test.mean()) < 0.05


def test_every_candidate_fits_and_outputs_probabilities():
    X, y = make_dataset()
    X_train, X_test, y_train, _ = split_data(X, y)
    for name, pipeline in build_candidates().items():
        proba = pipeline.fit(X_train, y_train).predict_proba(X_test)[:, 1]
        assert ((proba >= 0) & (proba <= 1)).all(), name
