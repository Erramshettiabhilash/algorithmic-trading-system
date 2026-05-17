"""Smoke tests for the AI quant research platform scaffold."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from data.pipeline import align_time_index, normalize_ohlcv_columns
from evaluation.metrics import information_coefficient, max_drawdown, sharpe_ratio
from explainability import ModelCard
from features import FeatureConfig
from optimization import XGBoostSearchSpace
from rl import TradingAction, TradingEnvironmentConfig
from visualization.research_charts import prepare_equity_curve


def test_required_ai_quant_directories_exist() -> None:
    required = [
        "data",
        "features",
        "models",
        "evaluation",
        "optimization",
        "rl",
        "explainability",
        "visualization",
        "notebooks",
        "tests",
        "results",
    ]

    for directory in required:
        assert Path(directory).is_dir()


def test_market_data_pipeline_normalizes_schema_and_timezone() -> None:
    raw = pd.DataFrame(
        {
            "Open": [100.0, 101.0],
            "High": [102.0, 103.0],
            "Low": [99.0, 100.0],
            "Close": [101.0, 102.0],
            "Volume": [1000, 1100],
        },
        index=pd.to_datetime(["2024-01-02", "2024-01-01"]),
    )

    normalized = normalize_ohlcv_columns(raw)
    aligned = align_time_index(normalized)

    assert list(aligned.columns) == ["open", "high", "low", "close", "volume"]
    assert aligned.index.is_monotonic_increasing
    assert aligned.index.tz is not None


def test_starter_research_components_are_constructible() -> None:
    feature_config = FeatureConfig()
    search_space = XGBoostSearchSpace()
    environment_config = TradingEnvironmentConfig()
    card = ModelCard(
        name="xgboost_direction",
        target="next_period_direction",
        horizon="1D",
        intended_use="research signal ranking",
        main_risks=("lookahead bias", "regime instability"),
    )

    assert feature_config.return_windows[0] == 1
    assert search_space.max_depth == (2, 8)
    assert environment_config.initial_cash == 100_000.0
    assert TradingAction.HOLD == 1
    assert "lookahead bias" in card.main_risks


def test_core_finance_metrics_behave_sensibly() -> None:
    returns = pd.Series([0.01, -0.005, 0.002, 0.004])
    equity = prepare_equity_curve(returns, initial_capital=100.0)
    predictions = pd.Series([0.3, 0.1, -0.2, 0.4])
    realized = pd.Series([0.02, 0.00, -0.01, 0.03])

    assert equity.iloc[0] > 100.0
    assert sharpe_ratio(returns) != 0.0
    assert max_drawdown(equity) <= 0.0
    assert information_coefficient(predictions, realized) > 0.0
