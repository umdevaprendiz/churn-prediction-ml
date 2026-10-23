"""Gera os graficos e metricas de avaliacao dos modelos em `docs/figures/`.

Usa a mesma divisao treino/teste do `src.train`. O limiar de decisao
"otimo" e escolhido com predicoes de validacao cruzada no TREINO (nunca no
teste), para que as metricas do teste continuem honestas.

Uso:
    python -m src.evaluate
"""

from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import cross_val_predict

from src import viz
from src.data_prep import DATA_PATH_DEFAULT, prepare
from src.train import RANDOM_STATE, build_candidates, split_data

FIG_DIR = "docs/figures"
METRICS_PATH = "docs/metrics.json"

MODEL_LABELS = {
    "logistic_regression": "Regressão Logística",
    "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting",
}
MODEL_COLORS = {
    "logistic_regression": viz.BLUE,
    "random_forest": viz.ORANGE,
    "gradient_boosting": viz.AQUA,
}
FEATURE_LABELS = {
    "tenure_months": "Tempo de casa (meses)",
    "monthly_charges": "Mensalidade",
    "total_charges": "Total cobrado",
    "num_support_calls": "Ligações ao suporte",
    "contract_type": "Tipo de contrato",
    "internet_service": "Serviço de internet",
    "tech_support": "Suporte técnico",
    "payment_method": "Forma de pagamento",
    "has_dependents": "Tem dependentes",
    "senior_citizen": "Idoso",
}


def fit_models():
    X, y = prepare(DATA_PATH_DEFAULT)
    X_train, X_test, y_train, y_test = split_data(X, y)
    models = build_candidates()
    for pipeline in models.values():
        pipeline.fit(X_train, y_train)
    return models, X_train, X_test, y_train, y_test


def best_f1_threshold(model, X_train, y_train) -> float:
    """Limiar que maximiza o F1 em predicoes de validacao cruzada (so treino)."""
    proba = cross_val_predict(model, X_train, y_train, cv=5, method="predict_proba")[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_train, proba)
    f1 = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-9, None)
    return float(thresholds[np.argmax(f1)])


def plot_model_comparison(metrics: dict) -> None:
    keys = [("roc_auc", "ROC-AUC"), ("accuracy", "Accuracy"), ("precision", "Precision"),
            ("recall", "Recall"), ("f1", "F1")]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    width = 0.8 / len(metrics)
    x = np.arange(len(keys))
    for i, (name, m) in enumerate(metrics.items()):
        values = [m[k] for k, _ in keys]
        bars = ax.bar(x + i * width - 0.4 + width / 2, values, width - 0.04,
                      color=MODEL_COLORS[name], label=MODEL_LABELS[name])
        ax.bar_label(bars, labels=[f"{v:.2f}" for v in values], padding=3,
                     color=viz.INK_SECONDARY, fontsize=8.5)
    ax.set_xticks(x, [label for _, label in keys])
    ax.set_ylim(0, 1.0)
    ax.grid(axis="x", visible=False)
    ax.set_title("Comparação dos modelos no conjunto de teste", pad=30)
    ax.legend(loc="lower left", ncol=3, bbox_to_anchor=(0, 1.0))
    viz.save(fig, f"{FIG_DIR}/model_comparison.png")


def plot_roc_pr(models, X_test, y_test) -> None:
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(11, 4.6))
    for name, model in models.items():
        proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, proba)
        prec, rec, _ = precision_recall_curve(y_test, proba)
        label = MODEL_LABELS[name]
        ax_roc.plot(fpr, tpr, color=MODEL_COLORS[name],
                    label=f"{label} (AUC {roc_auc_score(y_test, proba):.3f})")
        ax_pr.plot(rec, prec, color=MODEL_COLORS[name],
                   label=f"{label} (AP {average_precision_score(y_test, proba):.3f})")
    ax_roc.plot([0, 1], [0, 1], color=viz.INK_MUTED, lw=1, ls="--", label="Aleatório")
    ax_pr.axhline(y_test.mean(), color=viz.INK_MUTED, lw=1, ls="--",
                  label=f"Taxa base ({y_test.mean():.0%})")
    ax_roc.set(title="Curva ROC", xlabel="Taxa de falsos positivos", ylabel="Taxa de verdadeiros positivos")
    ax_pr.set(title="Curva Precisão × Recall", xlabel="Recall", ylabel="Precisão", ylim=(0, 1.02))
    ax_roc.legend(loc="lower right")
    ax_pr.legend(loc="upper right")
    fig.tight_layout()
    viz.save(fig, f"{FIG_DIR}/roc_pr_curves.png")


def plot_confusion(model, X_test, y_test, threshold: float) -> dict:
    proba = model.predict_proba(X_test)[:, 1]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    summary = {}
    for ax, (title, thr) in zip(axes, [("Limiar padrão (0,50)", 0.5),
                                       (f"Limiar otimizado ({threshold:.2f})".replace(".", ","), threshold)]):
        pred = (proba >= thr).astype(int)
        disp = ConfusionMatrixDisplay.from_predictions(
            y_test, pred, display_labels=["Retido", "Churn"], cmap=viz.SEQUENTIAL,
            colorbar=False, ax=ax, values_format="d",
        )
        ax.grid(False)
        # numero branco sobre celula escura, escuro sobre celula clara
        half = disp.confusion_matrix.max() / 2
        for (i, j), text in np.ndenumerate(disp.text_):
            text.set(color="white" if disp.confusion_matrix[i, j] > half else viz.INK,
                     fontsize=15, fontweight="bold")
        ax.set_title(f"{title}\nPrecisão {precision_score(y_test, pred):.0%} · "
                     f"Recall {recall_score(y_test, pred):.0%} · F1 {f1_score(y_test, pred):.2f}",
                     fontsize=10.5, loc="left")
        ax.set(xlabel="Previsto", ylabel="Real")
        summary[f"{thr:.2f}"] = {
            "precision": precision_score(y_test, pred),
            "recall": recall_score(y_test, pred),
            "f1": f1_score(y_test, pred),
        }
    fig.tight_layout()
    viz.save(fig, f"{FIG_DIR}/confusion_matrix.png")
    return summary


def plot_threshold(model, X_test, y_test, threshold: float) -> None:
    proba = model.predict_proba(X_test)[:, 1]
    grid = np.linspace(0.05, 0.9, 86)
    prec = [precision_score(y_test, proba >= t, zero_division=0) for t in grid]
    rec = [recall_score(y_test, proba >= t) for t in grid]
    f1 = [f1_score(y_test, proba >= t) for t in grid]
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    ax.plot(grid, prec, color=viz.BLUE, label="Precisão")
    ax.plot(grid, rec, color=viz.ORANGE, label="Recall")
    ax.plot(grid, f1, color=viz.INK_SECONDARY, ls="--", lw=1.6, label="F1")
    ax.axvline(threshold, color=viz.INK_MUTED, lw=1)
    ax.text(threshold + 0.01, 0.04, f"limiar otimizado = {threshold:.2f}".replace(".", ","),
            color=viz.INK_SECONDARY, fontsize=9)
    ax.set(title="Trade-off precisão × recall por limiar de decisão",
           xlabel="Limiar de probabilidade para classificar como churn", ylim=(0, 1.02))
    viz.pct_axis(ax)
    ax.legend(loc="center right")
    viz.save(fig, f"{FIG_DIR}/threshold_analysis.png")


def plot_importance(model, X_test, y_test) -> dict:
    result = permutation_importance(
        model, X_test, y_test, scoring="roc_auc", n_repeats=20, random_state=RANDOM_STATE
    )
    imp = pd.DataFrame({"feature": X_test.columns, "mean": result.importances_mean,
                        "std": result.importances_std}).sort_values("mean")
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.barh([FEATURE_LABELS[f] for f in imp["feature"]], imp["mean"], xerr=imp["std"],
            color=viz.BLUE, error_kw={"ecolor": viz.INK_MUTED, "lw": 1, "capsize": 2})
    ax.grid(axis="y", visible=False)
    ax.set(title="Importância das variáveis (permutação)",
           xlabel="Queda no ROC-AUC ao embaralhar a variável")
    viz.save(fig, f"{FIG_DIR}/feature_importance.png")
    return imp.sort_values("mean", ascending=False).set_index("feature")["mean"].round(4).to_dict()


def plot_calibration_lift(model, X_test, y_test) -> pd.DataFrame:
    proba = model.predict_proba(X_test)[:, 1]
    frac_pos, mean_pred = calibration_curve(y_test, proba, n_bins=8, strategy="quantile")

    decile = pd.qcut(pd.Series(proba).rank(method="first"), 10, labels=False)
    table = pd.DataFrame({"decile": decile.values, "churn": y_test.values}).groupby("decile")["churn"].agg(
        ["mean", "sum", "count"])

    fig, (ax_cal, ax_lift) = plt.subplots(1, 2, figsize=(11, 4.6))
    ax_cal.plot([0, 1], [0, 1], color=viz.INK_MUTED, ls="--", lw=1, label="Calibração perfeita")
    ax_cal.plot(mean_pred, frac_pos, color=viz.BLUE, marker="o", ms=6, mfc=viz.SURFACE,
                mew=2, label="Modelo")
    ax_cal.set(title="Calibração das probabilidades", xlabel="Probabilidade prevista",
               ylabel="Fração real de churn", xlim=(0, 1), ylim=(0, 1))
    viz.pct_axis(ax_cal); viz.pct_axis(ax_cal, "x")
    ax_cal.legend(loc="upper left")

    colors = [viz.ORANGE if i >= 7 else viz.BLUE for i in table.index]
    bars = ax_lift.bar(table.index + 1, table["mean"], color=colors, width=0.75)
    ax_lift.bar_label(bars, labels=[f"{v:.0%}" for v in table["mean"]], padding=3,
                      color=viz.INK_SECONDARY, fontsize=8.5)
    ax_lift.axhline(y_test.mean(), color=viz.INK_MUTED, ls="--", lw=1)
    ax_lift.text(0.6, y_test.mean() + 0.015, f"média geral {y_test.mean():.0%}",
                 color=viz.INK_SECONDARY, fontsize=9)
    ax_lift.set(title="Taxa real de churn por decil de risco",
                xlabel="Decil de risco previsto (10 = maior risco)", ylabel="Taxa real de churn",
                xticks=range(1, 11))
    ax_lift.grid(axis="x", visible=False)
    viz.pct_axis(ax_lift)
    fig.tight_layout()
    viz.save(fig, f"{FIG_DIR}/calibration_lift.png")
    return table


def plot_score_distribution(model, X_test, y_test) -> None:
    proba = model.predict_proba(X_test)[:, 1]
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    bins = np.linspace(0, 1, 26)
    for cls in (0, 1):
        ax.hist(proba[y_test.values == cls], bins=bins, alpha=0.75, color=viz.CHURN_PALETTE[cls],
                label=viz.CHURN_LABELS[cls], edgecolor=viz.SURFACE, linewidth=1.5)
    ax.set(title="Separação das probabilidades previstas por classe real",
           xlabel="Probabilidade de churn prevista", ylabel="Clientes")
    ax.legend()
    viz.save(fig, f"{FIG_DIR}/score_distribution.png")


def main() -> None:
    viz.apply_style()
    os.makedirs(FIG_DIR, exist_ok=True)

    models, X_train, X_test, y_train, y_test = fit_models()

    metrics = {}
    for name, model in models.items():
        proba = model.predict_proba(X_test)[:, 1]
        pred = model.predict(X_test)
        metrics[name] = {
            "accuracy": float((pred == y_test).mean()),
            "precision": float(precision_score(y_test, pred)),
            "recall": float(recall_score(y_test, pred)),
            "f1": float(f1_score(y_test, pred)),
            "roc_auc": float(roc_auc_score(y_test, proba)),
            "avg_precision": float(average_precision_score(y_test, proba)),
        }

    best_name = max(metrics, key=lambda n: metrics[n]["roc_auc"])
    best = models[best_name]
    threshold = best_f1_threshold(best, X_train, y_train)

    plot_model_comparison(metrics)
    plot_roc_pr(models, X_test, y_test)
    thr_summary = plot_confusion(best, X_test, y_test, threshold)
    plot_threshold(best, X_test, y_test, threshold)
    importance = plot_importance(best, X_test, y_test)
    deciles = plot_calibration_lift(best, X_test, y_test)
    plot_score_distribution(best, X_test, y_test)

    top3 = deciles.iloc[-3:]["sum"].sum() / deciles["sum"].sum()
    out = {
        "best_model": best_name,
        "optimized_threshold": threshold,
        "models": metrics,
        "threshold_comparison": thr_summary,
        "permutation_importance": importance,
        "top_3_deciles_capture_of_churn": float(top3),
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"Melhor modelo: {best_name}")
    print(f"Limiar otimizado (F1, via CV no treino): {threshold:.3f}")
    print(f"Os 3 decis de maior risco capturam {top3:.0%} dos churns reais")
    print(f"Figuras em {FIG_DIR}/ e métricas em {METRICS_PATH}")


if __name__ == "__main__":
    main()
