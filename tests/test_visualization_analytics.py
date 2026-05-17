"""Tests for Step 12 visualization and analytics helpers."""

from __future__ import annotations

import numpy as np
import pytest

from analytics.reporting import option_analytics_snapshot, option_chain_analytics
from utils.instruments import EuropeanOption, OptionType
from utils.market import MarketEnvironment
from visualization.charts import (
    plot_option_chain_greek_bars,
    plot_option_payoff,
    plot_option_price_heatmap,
    plot_option_price_vs_spot,
)


def _market() -> MarketEnvironment:
    return MarketEnvironment(spot=100.0, risk_free_rate=0.05, volatility=0.2)


def _call(strike: float = 100.0) -> EuropeanOption:
    return EuropeanOption(strike=strike, maturity=1.0, option_type=OptionType.CALL)


def test_option_analytics_snapshot_contains_price_and_greeks() -> None:
    snapshot = option_analytics_snapshot(option=_call(), market=_market())

    assert snapshot.price == pytest.approx(10.4506, abs=1e-4)
    assert snapshot.greeks.delta == pytest.approx(0.6368, abs=1e-4)
    assert snapshot.greeks.vega > 0.0


def test_option_chain_analytics_returns_sorted_table() -> None:
    options = [_call(110.0), _call(90.0)]
    table = option_chain_analytics(options=options, market=_market())

    assert list(table["strike"]) == [90.0, 110.0]
    assert {"price", "delta", "gamma", "vega", "theta", "rho"}.issubset(table.columns)


def test_option_chain_analytics_rejects_empty_options() -> None:
    with pytest.raises(ValueError, match="at least one"):
        option_chain_analytics(options=[], market=_market())


def test_plot_option_payoff_returns_figure(tmp_path) -> None:
    terminal_prices = np.linspace(50.0, 150.0, 21)
    fig = plot_option_payoff(
        terminal_prices=terminal_prices,
        option=_call(),
        premium=10.0,
        output_path=str(tmp_path / "payoff.png"),
    )

    assert len(fig.axes) == 1


def test_plot_option_price_vs_spot_returns_figure(tmp_path) -> None:
    fig = plot_option_price_vs_spot(
        spots=np.linspace(80.0, 120.0, 11),
        option=_call(),
        market=_market(),
        output_path=str(tmp_path / "price_vs_spot.png"),
    )

    assert len(fig.axes) == 1


def test_plot_option_price_heatmap_returns_figure(tmp_path) -> None:
    fig = plot_option_price_heatmap(
        spots=np.linspace(80.0, 120.0, 5),
        volatilities=np.linspace(0.1, 0.4, 4),
        option=_call(),
        market=_market(),
        output_path=str(tmp_path / "heatmap.png"),
    )

    assert len(fig.axes) == 2


def test_plot_option_chain_greek_bars_returns_figure(tmp_path) -> None:
    table = option_chain_analytics(options=[_call(90.0), _call(110.0)], market=_market())
    fig = plot_option_chain_greek_bars(
        analytics_table=table,
        greek="delta",
        output_path=str(tmp_path / "delta_bars.png"),
    )

    assert len(fig.axes) == 1


def test_plot_option_chain_greek_bars_requires_columns() -> None:
    table = option_chain_analytics(options=[_call()], market=_market()).drop(columns=["delta"])

    with pytest.raises(ValueError, match="missing required columns"):
        plot_option_chain_greek_bars(analytics_table=table, greek="delta")

