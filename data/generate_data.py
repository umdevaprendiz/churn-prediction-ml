"""Gera um dataset sintetico de churn de clientes (estilo telecom).

Nao usa nenhuma fonte externa: os dados sao gerados localmente com uma
semente fixa (reprodutivel), mas com correlacoes realistas entre as
features e o alvo (churn), para simular um cenario real de negocio.

Uso:
    python data/generate_data.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

N_ROWS = 3000
SEED = 42


def generate(n_rows: int = N_ROWS, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    tenure_months = rng.integers(0, 72, size=n_rows)
    monthly_charges = np.round(rng.normal(65, 25, size=n_rows).clip(15, 150), 2)
    contract_type = rng.choice(
        ["month-to-month", "one_year", "two_year"], size=n_rows, p=[0.55, 0.25, 0.20]
    )
    internet_service = rng.choice(
        ["fiber", "dsl", "none"], size=n_rows, p=[0.45, 0.35, 0.20]
    )
    tech_support = rng.choice(["yes", "no"], size=n_rows, p=[0.35, 0.65])
    payment_method = rng.choice(
        ["credit_card", "bank_transfer", "electronic_check", "mailed_check"],
        size=n_rows,
    )
    num_support_calls = rng.poisson(1.5, size=n_rows)
    has_dependents = rng.choice(["yes", "no"], size=n_rows, p=[0.3, 0.7])
    senior_citizen = rng.choice([0, 1], size=n_rows, p=[0.85, 0.15])

    total_charges = np.round(monthly_charges * np.maximum(tenure_months, 1) *
                              rng.normal(1.0, 0.05, size=n_rows), 2)

    # Probabilidade "verdadeira" de churn, combinando os fatores de forma
    # realista (contrato mensal, poucas ligacoes de suporte tecnico, tenure
    # baixo e muitas ligacoes de suporte aumentam o risco de churn).
    logit = (
        -1.5
        + 1.8 * (contract_type == "month-to-month")
        - 1.0 * (contract_type == "two_year")
        + 0.35 * (tech_support == "no")
        + 0.25 * num_support_calls
        - 0.03 * tenure_months
        + 0.01 * (monthly_charges - 65)
        + 0.5 * (payment_method == "electronic_check")
        + 0.2 * senior_citizen
        - 0.3 * (has_dependents == "yes")
    )
    prob_churn = 1 / (1 + np.exp(-logit))
    churn = rng.binomial(1, prob_churn)

    df = pd.DataFrame(
        {
            "tenure_months": tenure_months,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "contract_type": contract_type,
            "internet_service": internet_service,
            "tech_support": tech_support,
            "payment_method": payment_method,
            "num_support_calls": num_support_calls,
            "has_dependents": has_dependents,
            "senior_citizen": senior_citizen,
            "churn": churn,
        }
    )

    # Injeta alguns valores faltantes e duplicatas de proposito, para que o
    # pipeline de limpeza (src/data_prep.py) tenha algo real para tratar.
    missing_idx = rng.choice(n_rows, size=int(n_rows * 0.03), replace=False)
    df.loc[missing_idx, "total_charges"] = np.nan

    dup_idx = rng.choice(n_rows, size=int(n_rows * 0.01), replace=False)
    df = pd.concat([df, df.loc[dup_idx]], ignore_index=True)

    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


if __name__ == "__main__":
    df = generate()
    df.to_csv("data/churn.csv", index=False)
    print(f"Gerado data/churn.csv com {len(df)} linhas.")
    print(df["churn"].value_counts(normalize=True))
