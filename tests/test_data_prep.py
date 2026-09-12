from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_prep import CATEGORICAL_FEATURES, NUMERIC_FEATURES, clean_data, split_features_target


def make_raw_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "tenure_months": [1, 12, 12, np.nan],
            "monthly_charges": [50.0, 80.0, 80.0, 60.0],
            "total_charges": [50.0, 960.0, 960.0, np.nan],
            "num_support_calls": [0, 1, 1, 2],
            "contract_type": ["month-to-month", "two_year", "two_year", "one_year"],
            "internet_service": ["fiber", "dsl", "dsl", "none"],
            "tech_support": ["no", "yes", "yes", "no"],
            "payment_method": ["credit_card", "mailed_check", "mailed_check", "bank_transfer"],
            "has_dependents": ["no", "yes", "yes", "no"],
            "senior_citizen": [0, 0, 0, 1],
            "churn": [1, 0, 0, 1],
        }
    )


def test_clean_data_drops_duplicates():
    df = make_raw_df()
    result = clean_data(df)
    assert len(result) == 3


def test_clean_data_fills_missing_numeric():
    df = make_raw_df().drop_duplicates()
    result = clean_data(df)
    assert result[NUMERIC_FEATURES].isna().sum().sum() == 0


def test_split_features_target():
    df = clean_data(make_raw_df())
    X, y = split_features_target(df)
    assert list(X.columns) == NUMERIC_FEATURES + CATEGORICAL_FEATURES
    assert y.name == "churn"
    assert len(X) == len(y)
