"""Testes do gerador de dados sinteticos (data/generate_data.py)."""

from __future__ import annotations

import importlib.util
import pathlib

import pandas as pd
import pytest

_spec = importlib.util.spec_from_file_location(
    "generate_data", pathlib.Path(__file__).resolve().parents[1] / "data" / "generate_data.py"
)
generate_data = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(generate_data)

EXPECTED_COLUMNS = [
    "tenure_months", "monthly_charges", "total_charges", "contract_type", "internet_service",
    "tech_support", "payment_method", "num_support_calls", "has_dependents", "senior_citizen", "churn",
]


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    return generate_data.generate(n_rows=2000, seed=7)


def test_columns(df):
    assert sorted(df.columns) == sorted(EXPECTED_COLUMNS)


def test_row_count_includes_injected_duplicates(df):
    assert len(df) == 2000 + int(2000 * 0.01)


def test_same_seed_is_reproducible():
    a = generate_data.generate(n_rows=300, seed=1)
    b = generate_data.generate(n_rows=300, seed=1)
    pd.testing.assert_frame_equal(a, b)


def test_different_seeds_differ():
    a = generate_data.generate(n_rows=300, seed=1)
    b = generate_data.generate(n_rows=300, seed=2)
    assert not a.equals(b)


def test_has_injected_missing_values_and_duplicates(df):
    assert df["total_charges"].isna().sum() == int(2000 * 0.03)
    assert df.duplicated().sum() >= 1
    assert df.drop(columns="total_charges").isna().sum().sum() == 0


def test_churn_is_binary_and_moderately_imbalanced(df):
    assert set(df["churn"].unique()) == {0, 1}
    assert 0.2 < df["churn"].mean() < 0.45


@pytest.mark.parametrize(
    "column, low, high",
    [("tenure_months", 0, 71), ("monthly_charges", 15, 150), ("num_support_calls", 0, 15),
     ("senior_citizen", 0, 1)],
)
def test_numeric_ranges(df, column, low, high):
    assert df[column].min() >= low
    assert df[column].max() <= high


@pytest.mark.parametrize(
    "column, allowed",
    [("contract_type", {"month-to-month", "one_year", "two_year"}),
     ("internet_service", {"fiber", "dsl", "none"}),
     ("tech_support", {"yes", "no"}),
     ("has_dependents", {"yes", "no"}),
     ("payment_method", {"credit_card", "bank_transfer", "electronic_check", "mailed_check"})],
)
def test_categorical_values(df, column, allowed):
    assert set(df[column].unique()) <= allowed


def test_planted_signal_month_to_month_churns_more(df):
    rates = df.groupby("contract_type")["churn"].mean()
    assert rates["month-to-month"] > rates["one_year"] > rates["two_year"]


def test_planted_signal_tenure_is_negatively_related_to_churn(df):
    assert df["tenure_months"].corr(df["churn"]) < 0
