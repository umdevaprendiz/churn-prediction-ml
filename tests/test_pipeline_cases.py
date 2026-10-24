"""Casos de borda da limpeza de dados, da predicao em lote e da exportacao para BI."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import export_for_bi, predict
from src.data_prep import CATEGORICAL_FEATURES, NUMERIC_FEATURES, clean_data, prepare
from src.insights import PERSONAS


def raw_df(**overrides) -> pd.DataFrame:
    row = {
        "tenure_months": 10, "monthly_charges": 50.0, "total_charges": 500.0, "num_support_calls": 1,
        "contract_type": "one_year", "internet_service": "dsl", "tech_support": "yes",
        "payment_method": "credit_card", "has_dependents": "no", "senior_citizen": 0, "churn": 0,
    }
    row.update(overrides)
    return pd.DataFrame([row])


# ------------------------------------------------------------ limpeza
def test_clean_data_is_idempotent():
    df = pd.concat([raw_df(), raw_df(total_charges=np.nan, tenure_months=20)], ignore_index=True)
    once = clean_data(df)
    pd.testing.assert_frame_equal(once, clean_data(once))


def test_clean_data_uses_median_not_mean_for_imputation():
    df = pd.concat(
        [raw_df(total_charges=100.0), raw_df(total_charges=200.0, tenure_months=2),
         raw_df(total_charges=9000.0, tenure_months=3), raw_df(total_charges=np.nan, tenure_months=4)],
        ignore_index=True,
    )
    assert clean_data(df).loc[3, "total_charges"] == 200.0


def test_clean_data_keeps_row_order_and_resets_index():
    df = pd.concat([raw_df(tenure_months=1), raw_df(tenure_months=1), raw_df(tenure_months=2)],
                   ignore_index=True)
    out = clean_data(df)
    assert list(out.index) == [0, 1]
    assert list(out["tenure_months"]) == [1, 2]


def test_clean_data_does_not_touch_categorical_columns():
    df = raw_df()
    out = clean_data(df)
    assert out[CATEGORICAL_FEATURES].equals(df[CATEGORICAL_FEATURES])


def test_clean_data_with_no_missing_values_changes_nothing():
    df = pd.concat([raw_df(), raw_df(tenure_months=33)], ignore_index=True)
    pd.testing.assert_frame_equal(clean_data(df), df)


@pytest.mark.parametrize("column", NUMERIC_FEATURES)
def test_every_numeric_column_is_imputed(column):
    df = pd.concat([raw_df(), raw_df(tenure_months=5), raw_df(tenure_months=9)], ignore_index=True)
    df.loc[0, column] = np.nan
    assert clean_data(df)[column].isna().sum() == 0


def test_prepare_on_real_csv_returns_aligned_clean_data():
    X, y = prepare("data/churn.csv")
    assert len(X) == len(y) == 3000
    assert X.isna().sum().sum() == 0
    assert set(y.unique()) == {0, 1}
    assert "churn" not in X.columns


# ------------------------------------------------------ predicao em lote
def test_batch_predict_from_csv(tmp_path):
    path = tmp_path / "novos.csv"
    pd.DataFrame(PERSONAS.values()).to_csv(path, index=False)
    out = predict.predict(str(path))
    assert len(out) == len(PERSONAS)
    assert set(out["churn_prediction"].unique()) <= {0, 1}
    assert out["churn_probability"].between(0, 1).all()


def test_batch_predict_keeps_original_columns(tmp_path):
    path = tmp_path / "novos.csv"
    pd.DataFrame(PERSONAS.values()).to_csv(path, index=False)
    out = predict.predict(str(path))
    assert set(NUMERIC_FEATURES + CATEGORICAL_FEATURES) <= set(out.columns)


def test_batch_predict_fails_clearly_when_column_is_missing(tmp_path):
    path = tmp_path / "ruim.csv"
    pd.DataFrame(PERSONAS.values()).drop(columns="contract_type").to_csv(path, index=False)
    with pytest.raises(KeyError):
        predict.predict(str(path))


# --------------------------------------------------------- export p/ BI
@pytest.fixture(scope="module")
def scored() -> pd.DataFrame:
    return export_for_bi.score_customers()


def test_scored_has_prediction_columns(scored):
    assert {"churn_prediction", "churn_probability"} <= set(scored.columns)


def test_scored_probability_is_valid_and_matches_prediction(scored):
    assert scored["churn_probability"].between(0, 1).all()
    assert (scored["churn_prediction"] == (scored["churn_probability"] >= 0.5).astype(int)).all()


def test_scored_has_no_missing_values_and_no_duplicates(scored):
    assert scored.isna().sum().sum() == 0
    assert not scored.duplicated().any()


def test_model_beats_the_trivial_baseline_on_all_customers(scored):
    baseline = 1 - scored["churn"].mean()  # sempre prever "nao cancela"
    assert (scored["churn_prediction"] == scored["churn"]).mean() > baseline


def test_riskier_customers_actually_churn_more(scored):
    top = scored.nlargest(300, "churn_probability")["churn"].mean()
    bottom = scored.nsmallest(300, "churn_probability")["churn"].mean()
    assert top > 2 * bottom
