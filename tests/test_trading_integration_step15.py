"""Tests for Step 15 trading integration and portfolio evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from trading import (
    ExecutionConfig,
    PositionSizingConfig,
    SignalConfig,
    construct_long_short_weights,
    evaluate_trading_performance,
    generate_directional_signals,
    generate_probability_signals,
    profit_factor,
    risk_adjusted_position_sizing,
    simulate_multi_asset_portfolio,
    simulate_signal_strategy,
)


def _index(rows: int = 30) -> pd.DatetimeIndex:
    return pd.date_range("2024-01-01", periods=rows, freq="B", tz="UTC")


def test_directional_and_probability_signals() -> None:
    predictions = pd.Series([-0.02, -0.001, 0.0, 0.004, 0.02], index=_index(5))
    signals = generate_directional_signals(
        predictions,
        SignalConfig(long_threshold=0.003, short_threshold=-0.003, neutral_zone=0.0),
    )
    probabilities = pd.Series([0.2, 0.48, 0.51, 0.8], index=_index(4))
    probability_signals = generate_probability_signals(probabilities)

    assert signals.tolist() == [-1.0, 0.0, 0.0, 1.0, 1.0]
    assert probability_signals.tolist() == [-1.0, 0.0, 0.0, 1.0]


def test_risk_adjusted_position_sizing_caps_positions() -> None:
    index = _index(40)
    signals = pd.Series([1.0, -1.0] * 20, index=index)
    returns = pd.Series(np.linspace(-0.01, 0.01, 40), index=index)
    positions = risk_adjusted_position_sizing(
        signals,
        returns,
        PositionSizingConfig(max_position=0.5, volatility_target=0.01, volatility_window=5),
    )

    assert positions.abs().max() <= 0.5
    assert positions.index.equals(index)


def test_construct_long_short_weights_balances_gross_exposure() -> None:
    index = _index(3)
    predictions = pd.DataFrame(
        {
            "asset_a": [0.10, 0.02, -0.03],
            "asset_b": [0.04, -0.01, 0.08],
            "asset_c": [-0.03, 0.06, 0.01],
        },
        index=index,
    )

    weights = construct_long_short_weights(predictions, gross_leverage=1.0, top_n=1)

    assert weights.abs().sum(axis=1).eq(1.0).all()
    assert weights.sum(axis=1).abs().lt(1e-12).all()


def test_single_asset_signal_strategy_accounts_for_costs_and_metrics() -> None:
    index = _index(20)
    predictions = pd.Series([0.01, -0.01, 0.02, -0.02] * 5, index=index)
    realized = pd.Series([0.005, -0.004, 0.006, -0.003] * 5, index=index)

    result = simulate_signal_strategy(
        predictions,
        realized,
        execution_config=ExecutionConfig(transaction_cost_bps=2.0, slippage_bps=1.0),
    )

    assert result.performance.observations == 20
    assert result.performance.profit_factor > 1.0
    assert result.turnover.iloc[0] == 1.0
    assert result.equity_curve.iloc[-1] > 1.0


def test_multi_asset_portfolio_simulation() -> None:
    index = _index(10)
    weights = pd.DataFrame(
        {
            "asset_a": [0.5, 0.5, 0.0, 0.0, 0.4, 0.4, 0.2, 0.2, 0.0, 0.0],
            "asset_b": [-0.5, -0.5, 0.0, 0.0, -0.4, -0.4, -0.2, -0.2, 0.0, 0.0],
        },
        index=index,
    )
    returns = pd.DataFrame(
        {
            "asset_a": [0.01, 0.02, -0.01, 0.0, 0.01, 0.02, 0.01, -0.01, 0.0, 0.01],
            "asset_b": [-0.01, -0.02, 0.01, 0.0, -0.01, -0.02, 0.0, 0.01, 0.0, -0.01],
        },
        index=index,
    )

    result = simulate_multi_asset_portfolio(
        weights,
        returns,
        execution_config=ExecutionConfig(transaction_cost_bps=1.0),
    )

    assert result.performance.observations == 10
    assert result.positions.shape == weights.shape
    assert result.performance.total_return > 0.0


def test_trading_performance_metrics_and_profit_factor() -> None:
    returns = pd.Series([0.02, -0.01, 0.015, -0.005, 0.01], index=_index(5))
    benchmark = pd.Series([0.005] * 5, index=_index(5))
    turnover = pd.Series([1.0, 0.5, 0.0, 0.2, 0.1], index=_index(5))
    performance = evaluate_trading_performance(
        returns,
        benchmark_returns=benchmark,
        turnover=turnover,
        periods_per_year=252,
    )

    assert performance.cagr > 0.0
    assert performance.max_drawdown <= 0.0
    assert performance.average_turnover == turnover.mean()
    assert profit_factor(returns) > 1.0
