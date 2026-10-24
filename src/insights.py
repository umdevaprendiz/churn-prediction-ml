"""Graficos extras que traduzem o modelo para decisoes de negocio.

Complementa `src.evaluate` com:
  - perfis ("personas") de clientes e o risco previsto para cada um;
  - razoes de chances da Regressao Logistica (o que aumenta/diminui o risco);
  - cenarios "e se?" (tempo de casa x tipo de contrato, ligacoes x suporte);
  - curva de ganhos cumulativos (quantos churns pegamos contatando X% da base);
  - retorno financeiro (ilustrativo) de uma campanha de retencao;
  - curva de aprendizado e estabilidade da validacao cruzada.

Uso:
    python -m src.insights
"""

from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, learning_curve

from src import viz
from src.data_prep import CATEGORICAL_FEATURES, DATA_PATH_DEFAULT, NUMERIC_FEATURES, prepare
from src.evaluate import FEATURE_LABELS, FIG_DIR, MODEL_COLORS, MODEL_LABELS, fit_models
from src.train import RANDOM_STATE, build_candidates

INSIGHTS_PATH = "docs/insights.json"

# Premissas ILUSTRATIVAS da campanha de retencao (ajuste para o seu negocio).
CAMPAIGN = {
    "cost_per_contact": 60.0,  # R$ por cliente abordado (ligacao + oferta/desconto)
    "success_rate": 0.30,  # % dos churners abordados que ficam
    "months_saved": 12,  # meses de mensalidade preservados por cliente retido
}

PERSONAS: dict[str, dict] = {
    "Recém-chegado insatisfeito": dict(
        tenure_months=2, monthly_charges=95.0, total_charges=190.0, num_support_calls=4,
        contract_type="month-to-month", internet_service="fiber", tech_support="no",
        payment_method="electronic_check", has_dependents="no", senior_citizen=1),
    "Muitas ligações ao suporte": dict(
        tenure_months=12, monthly_charges=80.0, total_charges=960.0, num_support_calls=6,
        contract_type="month-to-month", internet_service="fiber", tech_support="no",
        payment_method="electronic_check", has_dependents="no", senior_citizen=0),
    "Mês a mês veterano": dict(
        tenure_months=36, monthly_charges=70.0, total_charges=2520.0, num_support_calls=1,
        contract_type="month-to-month", internet_service="fiber", tech_support="no",
        payment_method="bank_transfer", has_dependents="no", senior_citizen=0),
    "Contrato anual médio": dict(
        tenure_months=18, monthly_charges=65.0, total_charges=1170.0, num_support_calls=1,
        contract_type="one_year", internet_service="dsl", tech_support="yes",
        payment_method="credit_card", has_dependents="yes", senior_citizen=0),
    "Plano básico sem internet": dict(
        tenure_months=24, monthly_charges=20.0, total_charges=480.0, num_support_calls=1,
        contract_type="one_year", internet_service="none", tech_support="no",
        payment_method="mailed_check", has_dependents="no", senior_citizen=0),
    "Família estável": dict(
        tenure_months=48, monthly_charges=75.0, total_charges=3600.0, num_support_calls=1,
        contract_type="two_year", internet_service="fiber", tech_support="yes",
        payment_method="bank_transfer", has_dependents="yes", senior_citizen=0),
    "Cliente fiel": dict(
        tenure_months=60, monthly_charges=55.0, total_charges=3300.0, num_support_calls=0,
        contract_type="two_year", internet_service="dsl", tech_support="yes",
        payment_method="credit_card", has_dependents="yes", senior_citizen=0),
}

VALUE_LABELS = {
    "month-to-month": "mês a mês", "one_year": "1 ano", "two_year": "2 anos",
    "fiber": "fibra", "dsl": "DSL", "none": "sem internet",
    "yes": "sim", "no": "não",
    "credit_card": "cartão", "bank_transfer": "transferência",
    "electronic_check": "cheque eletrônico", "mailed_check": "cheque postal",
}


def score_personas(model) -> pd.Series:
    """Probabilidade de churn prevista para cada persona, ordenada da maior para a menor."""
    X = pd.DataFrame(PERSONAS.values(), index=list(PERSONAS))[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    return pd.Series(model.predict_proba(X)[:, 1], index=X.index).sort_values(ascending=False)


def cumulative_gains(y_true: np.ndarray, proba: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(% de clientes contatados, % dos churns capturados), contatando do maior risco ao menor."""
    order = np.argsort(-proba, kind="stable")
    captured = np.cumsum(y_true[order]) / max(y_true.sum(), 1)
    contacted = np.arange(1, len(y_true) + 1) / len(y_true)
    return np.insert(contacted, 0, 0.0), np.insert(captured, 0, 0.0)


def campaign_net_benefit(y_true: np.ndarray, proba: np.ndarray, monthly: np.ndarray,
                         campaign: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Beneficio liquido (R$) de abordar os k clientes de maior risco, para k = 0..n."""
    c = campaign or CAMPAIGN
    order = np.argsort(-proba, kind="stable")
    saved = c["success_rate"] * y_true[order] * monthly[order] * c["months_saved"]
    net = np.cumsum(saved) - c["cost_per_contact"] * np.arange(1, len(y_true) + 1)
    contacted = np.arange(1, len(y_true) + 1) / len(y_true)
    return np.insert(contacted, 0, 0.0), np.insert(net, 0, 0.0)


# ---------------------------------------------------------------- graficos


def plot_personas(model) -> pd.Series:
    scores = score_personas(model).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    colors = [viz.ORANGE if p >= 0.5 else viz.BLUE for p in scores]
    bars = ax.barh(scores.index, scores.values, color=colors, height=0.62)
    ax.bar_label(bars, labels=[f"{p:.0%}" for p in scores], padding=4, color=viz.INK_SECONDARY)
    ax.axvline(0.32, color=viz.INK_MUTED, ls="--", lw=1)
    ax.text(0.33, 0.97, "média da base (32%)", transform=ax.get_xaxis_transform(),
            color=viz.INK_SECONDARY, fontsize=9, va="top")
    ax.set(title="Risco de churn previsto para 7 perfis de cliente", xlim=(0, 1.02),
           xlabel="Probabilidade de churn prevista pelo modelo")
    ax.grid(axis="y", visible=False)
    viz.pct_axis(ax, "x")
    viz.save(fig, f"{FIG_DIR}/customer_personas.png")
    return scores.iloc[::-1]


def _feature_label(name: str) -> str:
    kind, rest = name.split("__", 1)
    if kind == "num":
        return f"{FEATURE_LABELS[rest]} (+1 desvio-padrão)"
    for feat in CATEGORICAL_FEATURES:
        if rest.startswith(feat + "_"):
            value = rest[len(feat) + 1:]
            return f"{FEATURE_LABELS[feat]}: {VALUE_LABELS.get(value, 'sim' if value == '1' else value)}"
    return rest


def logistic_odds_ratios(pipeline) -> pd.Series:
    """Razao de chances por variavel (exp do coeficiente), sem os niveis redundantes."""
    names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    coefs = pipeline.named_steps["model"].coef_[0]
    series = pd.Series(np.exp(coefs), index=names)
    # em variaveis binarias o nivel "nao"/0 e o espelho do "sim"/1
    redundant = [n for n in names if n.endswith(("_no", "_0"))]
    return series.drop(redundant)


def plot_odds_ratios(models) -> pd.Series:
    odds = logistic_odds_ratios(models["logistic_regression"]).sort_values()
    fig, ax = plt.subplots(figsize=(9.5, 6.6))
    logs = np.log(odds)
    ax.barh([_feature_label(n) for n in odds.index], logs,
            color=[viz.ORANGE if v > 0 else viz.BLUE for v in logs], height=0.65)
    ax.axvline(0, color=viz.INK_SECONDARY, lw=1)
    ticks = [0.25, 0.5, 1, 2, 4]
    ax.set_xticks(np.log(ticks), [f"{t:g}×" for t in ticks])
    for y, (v, o) in enumerate(zip(logs, odds)):
        ax.text(v + (0.04 if v > 0 else -0.04), y, f"{o:.2f}×", va="center",
                ha="left" if v > 0 else "right", fontsize=8.5, color=viz.INK_SECONDARY)
    ax.set(title="O que aumenta (laranja) ou reduz (azul) o risco de churn",
           xlabel="Razão de chances (Regressão Logística), mantendo as demais variáveis fixas",
           xlim=(logs.min() - 0.35, logs.max() + 0.3))
    ax.grid(axis="y", visible=False)
    viz.save(fig, f"{FIG_DIR}/odds_ratios.png")
    return odds


def _what_if(model, X: pd.DataFrame, **overrides) -> float:
    Xc = X.copy()
    for col, value in overrides.items():
        Xc[col] = value
    if "tenure_months" in overrides:  # mantem total_charges coerente com o tempo de casa
        Xc["total_charges"] = Xc["monthly_charges"] * np.maximum(Xc["tenure_months"], 1)
    return float(model.predict_proba(Xc)[:, 1].mean())


def plot_what_if(model, X: pd.DataFrame) -> None:
    fig, (ax_t, ax_c) = plt.subplots(1, 2, figsize=(12, 4.6))

    grid = np.arange(0, 72, 3)
    series = {"month-to-month": viz.ORANGE, "one_year": viz.BLUE, "two_year": viz.AQUA}
    for contract, color in series.items():
        ys = [_what_if(model, X, contract_type=contract, tenure_months=t) for t in grid]
        ax_t.plot(grid, ys, color=color, label=VALUE_LABELS[contract])
        ax_t.text(grid[-1] + 1, ys[-1], VALUE_LABELS[contract], color=viz.INK_SECONDARY,
                  va="center", fontsize=9)
    ax_t.set(title="E se o cliente ficar mais tempo?", xlabel="Tempo de casa (meses)",
             ylabel="Risco médio previsto", xlim=(0, 82), ylim=(0, 0.85))
    viz.pct_axis(ax_t)
    ax_t.legend(title="Contrato", loc="upper right")

    calls = np.arange(0, 8)
    for support, color, label in [("no", viz.ORANGE, "sem suporte"), ("yes", viz.BLUE, "com suporte")]:
        ys = [_what_if(model, X, tech_support=support, num_support_calls=n) for n in calls]
        ax_c.plot(calls, ys, color=color, marker="o", ms=5, mfc=viz.SURFACE, mew=1.8, label=label)
    ax_c.set(title="E se ligar mais vezes ao suporte?", xlabel="Ligações ao suporte",
             ylabel="Risco médio previsto", ylim=(0, 0.85))
    viz.pct_axis(ax_c)
    ax_c.legend(title="Suporte técnico", loc="upper left")
    fig.tight_layout()
    viz.save(fig, f"{FIG_DIR}/what_if_scenarios.png")


def plot_gains(model, X_test, y_test) -> dict:
    proba = model.predict_proba(X_test)[:, 1]
    y = y_test.to_numpy()
    contacted, captured = cumulative_gains(y, proba)
    ideal = np.minimum(contacted / y.mean(), 1.0)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(contacted, ideal, color=viz.INK_MUTED, ls=":", lw=1.5, label="Modelo ideal")
    ax.plot(contacted, captured, color=viz.BLUE, label="Modelo")
    ax.plot([0, 1], [0, 1], color=viz.INK_MUTED, ls="--", lw=1, label="Sem modelo (aleatório)")
    marks = {}
    for frac in (0.2, 0.3):
        i = int(round(frac * len(y)))
        marks[f"{frac:.0%}"] = float(captured[i])
        ax.plot(frac, captured[i], "o", color=viz.ORANGE, ms=8, mec=viz.SURFACE, mew=2)
        ax.annotate(f"{frac:.0%} da base → {captured[i]:.0%} dos churns", (frac, captured[i]),
                    textcoords="offset points", xytext=(14, -16), fontsize=9, color=viz.INK_SECONDARY)
    ax.set(title="Ganhos cumulativos: quantos churns pegamos abordando só os de maior risco",
           xlabel="% de clientes abordados (do maior para o menor risco)",
           ylabel="% dos churns capturados", xlim=(0, 1), ylim=(0, 1.02))
    viz.pct_axis(ax); viz.pct_axis(ax, "x")
    ax.legend(loc="lower right")
    viz.save(fig, f"{FIG_DIR}/cumulative_gains.png")
    return marks


def plot_campaign_roi(model, X_test, y_test) -> dict:
    proba = model.predict_proba(X_test)[:, 1]
    contacted, net = campaign_net_benefit(y_test.to_numpy(), proba, X_test["monthly_charges"].to_numpy())
    best = int(np.argmax(net))

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.fill_between(contacted, net, 0, where=net >= 0, color=viz.BLUE, alpha=0.18, linewidth=0)
    ax.fill_between(contacted, net, 0, where=net < 0, color=viz.ORANGE, alpha=0.25, linewidth=0)
    ax.plot(contacted, net, color=viz.BLUE)
    ax.axhline(0, color=viz.INK_SECONDARY, lw=1)
    ax.plot(contacted[best], net[best], "o", color=viz.ORANGE, ms=9, mec=viz.SURFACE, mew=2)
    ax.annotate(f"melhor ponto: abordar {contacted[best]:.0%} da base\nbenefício líquido ≈ R$ {net[best]:,.0f}"
                .replace(",", "."), (contacted[best], net[best]), textcoords="offset points",
                xytext=(16, -8), fontsize=9.5, color=viz.INK)
    c = CAMPAIGN
    ax.set(title="Retorno ilustrativo de uma campanha de retenção",
           xlabel="% de clientes abordados (do maior para o menor risco)\n"
                  f"Premissas: R$ {c['cost_per_contact']:.0f}/contato · {c['success_rate']:.0%} de sucesso · "
                  f"{c['months_saved']} meses de mensalidade preservados",
           ylabel="Benefício líquido (R$, 600 clientes do teste)", xlim=(0, 1))
    viz.pct_axis(ax, "x")
    from matplotlib.ticker import FuncFormatter
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"R$ {v:,.0f}".replace(",", ".")))
    viz.save(fig, f"{FIG_DIR}/campaign_roi.png")
    return {"best_fraction": float(contacted[best]), "best_net_benefit": float(net[best]),
            "assumptions": c}


def plot_learning_curve(X, y) -> None:
    pipeline = build_candidates()["logistic_regression"]
    sizes, train, val = learning_curve(
        clone(pipeline), X, y, cv=5, scoring="roc_auc", train_sizes=np.linspace(0.15, 1.0, 7),
        random_state=RANDOM_STATE, shuffle=True,
    )
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for scores, color, label in [(train, viz.ORANGE, "Treino"), (val, viz.BLUE, "Validação cruzada")]:
        m, s = scores.mean(axis=1), scores.std(axis=1)
        ax.fill_between(sizes, m - s, m + s, color=color, alpha=0.15, linewidth=0)
        ax.plot(sizes, m, color=color, marker="o", ms=5, mfc=viz.SURFACE, mew=1.8, label=label)
    ax.set(title="Curva de aprendizado (Regressão Logística)", xlabel="Clientes usados no treino",
           ylabel="ROC-AUC", ylim=(0.7, 0.9))
    ax.legend(loc="lower right")
    viz.save(fig, f"{FIG_DIR}/learning_curve.png")


def plot_cv_stability(X, y) -> dict:
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=RANDOM_STATE)
    rng = np.random.default_rng(RANDOM_STATE)
    results = {}
    fig, ax = plt.subplots(figsize=(8.5, 4))
    for i, (name, pipeline) in enumerate(build_candidates().items()):
        scores = cross_val_score(pipeline, X, y, cv=cv, scoring="roc_auc")
        results[name] = {"mean": float(scores.mean()), "std": float(scores.std())}
        ax.scatter(scores, i + rng.uniform(-0.12, 0.12, len(scores)), s=26, alpha=0.7,
                   color=MODEL_COLORS[name], linewidths=0)
        ax.plot([scores.mean()] * 2, [i - 0.28, i + 0.28], color=viz.INK, lw=2)
        ax.text(0.885, i, f"média {scores.mean():.3f}".replace(".", ","),
                va="center", fontsize=9, color=viz.INK_SECONDARY)
    ax.set_yticks(range(len(results)), [MODEL_LABELS[n] for n in results])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set(title="Estabilidade do ROC-AUC (5 folds × 3 repetições)", xlabel="ROC-AUC",
           xlim=(0.74, 0.93))
    viz.save(fig, f"{FIG_DIR}/cv_stability.png")
    return results


def main() -> None:
    viz.apply_style()
    os.makedirs(FIG_DIR, exist_ok=True)

    models, X_train, X_test, y_train, y_test = fit_models()
    X, y = prepare(DATA_PATH_DEFAULT)
    best = models["logistic_regression"]

    personas = plot_personas(best)
    odds = plot_odds_ratios(models)
    plot_what_if(best, X_test)
    gains = plot_gains(best, X_test, y_test)
    roi = plot_campaign_roi(best, X_test, y_test)
    plot_learning_curve(X, y)
    cv = plot_cv_stability(X, y)

    out = {
        "personas": personas.round(4).to_dict(),
        "odds_ratios": {_feature_label(k): round(float(v), 3) for k, v in odds.items()},
        "gains_captured_at": gains,
        "campaign": roi,
        "cv_roc_auc": cv,
    }
    with open(INSIGHTS_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"Perfil de maior risco: {personas.index[0]} ({personas.iloc[0]:.0%})")
    print(f"Perfil de menor risco: {personas.index[-1]} ({personas.iloc[-1]:.0%})")
    print(f"Ganhos: {gains}")
    print(f"Campanha: melhor ponto = abordar {roi['best_fraction']:.0%} da base")
    print(f"Figuras em {FIG_DIR}/ e dados em {INSIGHTS_PATH}")


if __name__ == "__main__":
    main()
