"""Treina e compara modelos de classificacao para prever churn de clientes.

Uso:
    python -m src.train
"""

from __future__ import annotations

import joblib
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data_prep import CATEGORICAL_FEATURES, NUMERIC_FEATURES, prepare

DATA_PATH = "data/churn.csv"
MODEL_PATH = "models/churn_model.joblib"
RANDOM_STATE = 42


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def build_candidates(preprocessor: ColumnTransformer | None = None) -> dict[str, Pipeline]:
    """Um Pipeline por modelo, cada um com o seu proprio pre-processador.

    Compartilhar a mesma instancia de `ColumnTransformer` entre pipelines faz
    um sobrescrever o ajuste do outro; por isso cada candidato recebe um clone.
    """
    base = preprocessor if preprocessor is not None else build_preprocessor()
    return {
        "logistic_regression": Pipeline(
            [
                ("preprocessor", clone(base)),
                ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("preprocessor", clone(base)),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300, max_depth=8, random_state=RANDOM_STATE
                    ),
                ),
            ]
        ),
        "gradient_boosting": Pipeline(
            [
                ("preprocessor", clone(base)),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        max_depth=3,
                        learning_rate=0.05,
                        max_iter=200,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def split_data(X, y):
    """Divisao treino/teste estratificada, compartilhada por train e evaluate."""
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)


def evaluate(pipeline: Pipeline, X_test, y_test) -> dict:
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }


def main() -> None:
    X, y = prepare(DATA_PATH)
    X_train, X_test, y_train, y_test = split_data(X, y)

    candidates = build_candidates()

    results = {}
    fitted = {}
    for name, pipeline in candidates.items():
        pipeline.fit(X_train, y_train)
        fitted[name] = pipeline
        results[name] = evaluate(pipeline, X_test, y_test)
        cv_auc = cross_val_score(
            clone(pipeline), X_train, y_train, cv=5, scoring="roc_auc"
        )
        results[name]["cv_roc_auc_mean"] = cv_auc.mean()
        results[name]["cv_roc_auc_std"] = cv_auc.std()

    print("=== Comparacao de modelos ===")
    for name, metrics in results.items():
        print(f"\n{name}:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.4f}")

    best_name = max(results, key=lambda name: results[name]["roc_auc"])
    best_pipeline = fitted[best_name]
    print(f"\nMelhor modelo: {best_name} (roc_auc={results[best_name]['roc_auc']:.4f})")

    print("\n=== Classification report (melhor modelo) ===")
    print(classification_report(y_test, best_pipeline.predict(X_test)))

    joblib.dump(best_pipeline, MODEL_PATH)
    print(f"Modelo salvo em {MODEL_PATH}")


if __name__ == "__main__":
    main()
