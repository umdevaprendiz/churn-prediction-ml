"""API para servir o modelo de churn: previsao individual e geracao do
relatorio de risco usado pelo dashboard.
"""

from __future__ import annotations

import joblib
import pandas as pd
from fastapi import FastAPI

from api.schemas import CustomerInput, PredictionOutput, ReportSummary
from src.data_prep import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from src.export_for_bi import OUTPUT_PATH, score_customers

MODEL_PATH = "models/churn_model.joblib"

app = FastAPI(title="Churn Prediction API")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionOutput)
def predict(customer: CustomerInput) -> PredictionOutput:
    """Recebe os dados de 1 cliente e retorna a previsao de churn."""
    model = joblib.load(MODEL_PATH)
    X = pd.DataFrame([customer.model_dump()])[NUMERIC_FEATURES + CATEGORICAL_FEATURES]

    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0, 1]

    return PredictionOutput(
        churn_prediction=int(prediction), churn_probability=float(probability)
    )


@app.post("/generate-report", response_model=ReportSummary)
def generate_report() -> ReportSummary:
    """Roda o scoring de todos os clientes e salva data/churn_scored.csv."""
    df = score_customers()
    df.to_csv(OUTPUT_PATH, index=False)
    return ReportSummary(rows=len(df), output_path=OUTPUT_PATH)
