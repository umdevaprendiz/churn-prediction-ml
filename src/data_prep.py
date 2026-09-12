"""Limpeza e preparacao dos dados de churn.

Funcoes puras (sem efeitos colaterais) que recebem e retornam DataFrames,
para que o pipeline inteiro seja facil de testar unitariamente.
"""

from __future__ import annotations

import pandas as pd

TARGET_COLUMN = "churn"

NUMERIC_FEATURES = [
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "num_support_calls",
]

CATEGORICAL_FEATURES = [
    "contract_type",
    "internet_service",
    "tech_support",
    "payment_method",
    "has_dependents",
    "senior_citizen",
]


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicatas e preenche valores faltantes nas colunas numericas."""
    df = df.drop_duplicates().reset_index(drop=True)

    for col in NUMERIC_FEATURES:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    return df


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    y = df[TARGET_COLUMN].copy()
    return X, y


def prepare(path: str) -> tuple[pd.DataFrame, pd.Series]:
    df = load_data(path)
    df = clean_data(df)
    return split_features_target(df)
