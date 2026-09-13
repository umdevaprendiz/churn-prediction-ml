"""API para servir o modelo de churn.

EXERCICIO: implemente os dois endpoints marcados com `raise
NotImplementedError`. Veja TASK_API.md para a descricao completa de cada
um. Rode `pytest tests/test_api.py -v` para conferir seu progresso - os
testes ja estao prontos e nao devem ser alterados.
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
    raise NotImplementedError


@app.post("/generate-report", response_model=ReportSummary)
def generate_report() -> ReportSummary:
    """Roda o scoring de todos os clientes e salva data/churn_scored.csv."""
    raise NotImplementedError
