"""Formatos de entrada/saida da API, validados automaticamente pelo FastAPI."""

from __future__ import annotations

from pydantic import BaseModel


class CustomerInput(BaseModel):
    tenure_months: int
    monthly_charges: float
    total_charges: float
    num_support_calls: int
    contract_type: str
    internet_service: str
    tech_support: str
    payment_method: str
    has_dependents: str
    senior_citizen: int


class PredictionOutput(BaseModel):
    churn_prediction: int
    churn_probability: float


class ReportSummary(BaseModel):
    rows: int
    output_path: str
