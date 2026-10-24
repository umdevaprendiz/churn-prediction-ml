"""Testes das funcoes de apoio de src/insights.py e src/evaluate.py."""

from __future__ import annotations

import numpy as np
import pytest

from src import evaluate, insights, viz
from src.data_prep import CATEGORICAL_FEATURES, NUMERIC_FEATURES


# --------------------------------------------------------------- personas
def test_personas_cover_all_model_features():
    for name, persona in insights.PERSONAS.items():
        assert set(persona) == set(NUMERIC_FEATURES + CATEGORICAL_FEATURES), name


def test_personas_total_charges_is_consistent_with_tenure():
    for name, p in insights.PERSONAS.items():
        expected = p["monthly_charges"] * max(p["tenure_months"], 1)
        assert p["total_charges"] == pytest.approx(expected, rel=0.05), name


def test_score_personas_returns_sorted_probabilities(fitted_logreg):
    model = fitted_logreg[0]
    scores = insights.score_personas(model)
    assert len(scores) == len(insights.PERSONAS)
    assert ((scores >= 0) & (scores <= 1)).all()
    assert scores.is_monotonic_decreasing


def test_risky_persona_scores_above_loyal_persona(fitted_logreg):
    scores = insights.score_personas(fitted_logreg[0])
    assert scores["Recém-chegado insatisfeito"] > scores["Cliente fiel"]


# ----------------------------------------------------------------- ganhos
def test_cumulative_gains_endpoints_and_monotonic():
    y = np.array([1, 0, 1, 0, 0, 1, 0, 0])
    proba = np.array([0.9, 0.1, 0.8, 0.3, 0.2, 0.7, 0.4, 0.05])
    contacted, captured = insights.cumulative_gains(y, proba)
    assert contacted[0] == 0 and captured[0] == 0
    assert contacted[-1] == 1 and captured[-1] == 1
    assert np.all(np.diff(captured) >= 0)


def test_cumulative_gains_perfect_model_captures_all_churn_first():
    y = np.array([1, 1, 0, 0, 0])
    proba = np.array([0.9, 0.8, 0.3, 0.2, 0.1])
    contacted, captured = insights.cumulative_gains(y, proba)
    assert captured[2] == 1.0  # os 2 primeiros (40% da base) ja pegam 100% dos churns


def test_cumulative_gains_random_scores_stay_near_diagonal():
    rng = np.random.default_rng(0)
    y = rng.binomial(1, 0.3, 5000)
    contacted, captured = insights.cumulative_gains(y, rng.random(5000))
    assert np.abs(captured - contacted).max() < 0.05


# ---------------------------------------------------------- retorno (ROI)
CAMP = {"cost_per_contact": 10.0, "success_rate": 0.5, "months_saved": 12}


def test_net_benefit_starts_at_zero():
    y = np.array([1, 0, 1])
    _, net = insights.campaign_net_benefit(y, np.array([0.9, 0.5, 0.1]), np.array([50.0] * 3), CAMP)
    assert net[0] == 0


def test_net_benefit_first_true_churner_pays_off():
    y = np.array([1, 0, 0])
    _, net = insights.campaign_net_benefit(y, np.array([0.9, 0.5, 0.1]), np.array([100.0] * 3), CAMP)
    assert net[1] == pytest.approx(0.5 * 100 * 12 - 10)


def test_net_benefit_is_negative_when_nobody_churns():
    y = np.zeros(4, dtype=int)
    _, net = insights.campaign_net_benefit(y, np.linspace(0.9, 0.1, 4), np.array([80.0] * 4), CAMP)
    assert net[-1] == pytest.approx(-40.0)


@pytest.mark.parametrize("success_rate", [0.1, 0.3, 0.6])
def test_higher_success_rate_never_lowers_benefit(success_rate):
    rng = np.random.default_rng(1)
    y = rng.binomial(1, 0.3, 200)
    proba, monthly = rng.random(200), rng.uniform(20, 120, 200)
    base = insights.campaign_net_benefit(y, proba, monthly, {**CAMP, "success_rate": success_rate})[1]
    better = insights.campaign_net_benefit(y, proba, monthly, {**CAMP, "success_rate": success_rate + 0.1})[1]
    assert np.all(better >= base - 1e-9)


# --------------------------------------------------------- razao de chances
def test_odds_ratios_are_positive_and_drop_mirror_levels(fitted_logreg):
    odds = insights.logistic_odds_ratios(fitted_logreg[0])
    assert (odds > 0).all()
    assert not any(n.endswith(("_no", "_0")) for n in odds.index)


def test_odds_ratio_direction_matches_planted_signal(fitted_logreg):
    odds = insights.logistic_odds_ratios(fitted_logreg[0])
    assert odds["cat__contract_type_month-to-month"] > 1
    assert odds["num__tenure_months"] < 1


def test_feature_labels_are_human_readable():
    assert "mês a mês" in insights._feature_label("cat__contract_type_month-to-month")
    assert "desvio-padrão" in insights._feature_label("num__tenure_months")


# ------------------------------------------------------------- limiar
def test_best_f1_threshold_is_a_valid_probability(fitted_logreg):
    model, X_train, _, y_train, _ = fitted_logreg
    thr = evaluate.best_f1_threshold(model, X_train, y_train)
    assert 0 < thr < 1


def test_best_f1_threshold_is_below_default_for_imbalanced_target(fitted_logreg):
    """Com ~35-40% de positivos, maximizar F1 pede um limiar <= 0,5 (favorece recall)."""
    model, X_train, _, y_train, _ = fitted_logreg
    assert evaluate.best_f1_threshold(model, X_train, y_train) <= 0.55


# ------------------------------------------------------------ figuras
def test_plots_write_png_files(tmp_path, monkeypatch, fitted_logreg):
    model, _, X_test, _, y_test = fitted_logreg
    monkeypatch.setattr(insights, "FIG_DIR", str(tmp_path))
    monkeypatch.setattr(evaluate, "FIG_DIR", str(tmp_path))
    viz.apply_style()

    insights.plot_personas(model)
    insights.plot_gains(model, X_test, y_test)
    insights.plot_campaign_roi(model, X_test, y_test)
    insights.plot_what_if(model, X_test)
    evaluate.plot_score_distribution(model, X_test, y_test)

    names = {p.name for p in tmp_path.glob("*.png")}
    assert names == {"customer_personas.png", "cumulative_gains.png", "campaign_roi.png",
                     "what_if_scenarios.png", "score_distribution.png"}
    assert all((tmp_path / n).stat().st_size > 5_000 for n in names)


# ---------------------------------------------------------------- viz
def test_pct_axis_formats_percentages():
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    viz.pct_axis(ax)
    assert ax.yaxis.get_major_formatter().format_data(0.5).startswith("50")
    plt.close(fig)
