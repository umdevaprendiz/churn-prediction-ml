"""Carrega o modelo treinado e prediz churn para novos clientes.

Uso:
    python -m src.predict caminho/para/novos_clientes.csv
"""

from __future__ import annotations

import sys

import joblib
import pandas as pd

from src.data_prep import CATEGORICAL_FEATURES, NUMERIC_FEATURES

MODEL_PATH = "models/churn_model.joblib"


def predict(input_path: str) -> pd.DataFrame:
    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(input_path)
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]

    df["churn_prediction"] = model.predict(X)
    df["churn_probability"] = model.predict_proba(X)[:, 1]
    return df


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Uso: python -m src.predict <caminho_csv>")

    result = predict(sys.argv[1])
    print(result[["churn_prediction", "churn_probability"]])
