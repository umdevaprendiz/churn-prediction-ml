"""Gera um CSV com os clientes + a previsao do modelo, pronto pra importar
no Power BI (ou qualquer outra ferramenta de BI).

Importante: aqui o modelo prediz em cima de TODOS os clientes, incluindo os
que foram usados no treino. Isso e proposital - o objetivo deste arquivo e
alimentar um dashboard de negocio ("quais clientes tem mais risco de
cancelar"), nao avaliar a qualidade do modelo (isso ja e feito em
`src/train.py`, com dados de teste separados).

Uso:
    python -m src.export_for_bi
"""

from __future__ import annotations

import joblib
import pandas as pd

from src.data_prep import CATEGORICAL_FEATURES, NUMERIC_FEATURES, clean_data, load_data

DATA_PATH = "data/churn.csv"
MODEL_PATH = "models/churn_model.joblib"
OUTPUT_PATH = "data/churn_scored.csv"


def score_customers() -> pd.DataFrame:
    """Carrega o modelo e retorna todos os clientes com a previsao anexada.

    Extraida do `main()` para poder ser reaproveitada pela API (endpoint
    /generate-report), sem duplicar a logica de scoring.
    """
    model = joblib.load(MODEL_PATH)
    df = clean_data(load_data(DATA_PATH))

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    df["churn_prediction"] = model.predict(X)
    df["churn_probability"] = model.predict_proba(X)[:, 1]
    return df


def main() -> None:
    df = score_customers()
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Gerado {OUTPUT_PATH} com {len(df)} linhas.")


if __name__ == "__main__":
    main()
