"""Tests for Step 16 visualization and analytics reports."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from analytics import generate_factor_analytics_report, generate_model_evaluation_report
from trading import TradingPerformance
from visualization import (
    plot_drawdown_curve,
    plot_equity_curve,
    plot_feature_interaction,
    plot_performance_dashboard,
    plot_portfolio_allocation,
    plot_prediction_vs_actual,
    plot_regime_classification,
    plot_rl_rewards,
    plot_shap_importance,
)


def _index(rows: int = 30) -> pd.DatetimeIndex:
    return pd.date_range("2024-01-01", periods=rows, freq="B", tz="UTC")


def _assert_file(path: Path) -> None:
    assert path.exists()
    assert path.stat().st_size > 0


def test_prediction_equity_drawdown_and_dashboard_charts(tmp_path: Path) -> None:
    index = _index()
    predictions = pd.Series([0.001 * ((i % 5) - 2) for i in range(30)], index=index)
    actual = pd.Series([0.002 * ((i % 4) - 1) for i in range(30)], index=index)
    returns = pd.Series([0.01, -0.004, 0.006, -0.002, 0.003] * 6, index=index)

    paths = [
        tmp_path / "prediction_vs_actual.png",
        tmp_path / "equity.png",
        tmp_path / "drawdown.png",
        tmp_path / "dashboard.png",
    ]

    plot_prediction_vs_actual(predictions, actual, paths[0])
    plot_equity_curve(returns, paths[1])
    plot_drawdown_curve(returns, paths[2])
    plot_performance_dashboard(returns, paths[3])

    for path in paths:
        _assert_file(path)


def test_shap_interaction_regime_and_rl_charts(tmp_path: Path) -> None:
    index = _index()
    importance = pd.Series(
        {"momentum": 0.20, "volatility": 0.15, "volume": 0.05},
        name="mean_abs_shap",
    )
    dependence = pd.DataFrame(
        {
            "feature_value": [0.1, 0.2, 0.3, 0.4],
            "shap_value": [-0.01, 0.00, 0.02, 0.03],
        }
    )
    price = pd.Series(range(100, 130), index=index)
    regimes = pd.Series(["ranging", "trending", "high_volatility"] * 10, index=index)
    rewards = pd.Series([0.1, -0.05, 0.2, 0.0, 0.15] * 6, index=range(30))

    paths = [
        tmp_path / "shap.png",
        tmp_path / "interaction.png",
        tmp_path / "regime.png",
        tmp_path / "rewards.png",
    ]

    plot_shap_importance(importance, paths[0])
    plot_feature_interaction(dependence, paths[1])
    plot_regime_classification(price, regimes, paths[2])
    plot_rl_rewards(rewards, paths[3], rolling_window=5)

    for path in paths:
        _assert_file(path)


def test_portfolio_allocation_chart_writes_html(tmp_path: Path) -> None:
    weights = pd.DataFrame(
        {
            "asset_a": [0.5, 0.4, 0.2],
            "asset_b": [0.3, 0.4, 0.5],
            "asset_c": [0.2, 0.2, 0.3],
        },
        index=_index(3),
    )
    output = tmp_path / "allocation.html"

    plot_portfolio_allocation(weights, output)

    _assert_file(output)
    assert "Portfolio Allocation" in output.read_text(encoding="utf-8")


def test_research_reports_write_markdown(tmp_path: Path) -> None:
    performance = TradingPerformance(
        total_return=0.12,
        cagr=0.10,
        sharpe=1.2,
        information_ratio=0.8,
        max_drawdown=-0.05,
        annualized_volatility=0.15,
        profit_factor=1.6,
        win_rate=0.55,
        average_turnover=0.2,
        observations=252,
    )
    exposure = pd.Series({"market": 0.2, "momentum": 0.4})
    risk = pd.Series({"asset_a": 0.6, "asset_b": 0.4})
    model_report = tmp_path / "model_report.md"
    factor_report = tmp_path / "factor_report.md"

    generate_model_evaluation_report(
        model_report,
        title="Model Evaluation Report",
        metrics=performance,
        notes=["Out-of-sample walk-forward results."],
    )
    generate_factor_analytics_report(
        factor_report,
        exposure=exposure,
        risk_contribution=risk,
    )

    _assert_file(model_report)
    _assert_file(factor_report)
    assert "Model Evaluation Report" in model_report.read_text(encoding="utf-8")
    assert "Portfolio Factor Exposure" in factor_report.read_text(encoding="utf-8")
