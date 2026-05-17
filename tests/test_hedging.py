"""Tests for the Step 9 dynamic hedging simulator."""

from __future__ import annotations

import numpy as np
import pytest

from analytics.greeks import Greeks
from simulations.hedging import (
    HedgePortfolioLeg,
    HedgingSimulator,
    delta_neutral_stock_position,
    gamma_vega_neutral_option_positions,
)
from utils.instruments import EuropeanOption, OptionType
from utils.market import MarketEnvironment
from visualization.charts import plot_hedging_error_distribution, plot_hedging_pnl


def _option() -> EuropeanOption:
    return EuropeanOption(strike=100.0, maturity=1.0, option_type=OptionType.CALL)


def _market() -> MarketEnvironment:
    return MarketEnvironment(spot=100.0, risk_free_rate=0.02, volatility=0.2)


def test_delta_neutral_stock_position() -> None:
    assert delta_neutral_stock_position(option_position=-1.0, option_delta=0.6) == 0.6
    assert delta_neutral_stock_position(option_position=2.0, option_delta=0.6) == -1.2


def test_simulate_price_paths_shape() -> None:
    simulator = HedgingSimulator(option=_option(), market=_market())

    paths = simulator.simulate_price_paths(n_paths=4, steps=12, seed=7)

    assert paths.shape == (4, 13)
    np.testing.assert_allclose(paths[:, 0], np.full(4, 100.0))
    assert np.all(paths > 0.0)


def test_delta_hedge_path_returns_accounting_vectors() -> None:
    simulator = HedgingSimulator(option=_option(), market=_market())
    path = np.array([100.0, 102.0, 101.0, 105.0])

    result = simulator.simulate_delta_hedge_path(path)

    assert result.times.shape == path.shape
    assert result.stock_prices.shape == path.shape
    assert result.option_values.shape == path.shape
    assert result.deltas.shape == path.shape
    assert result.stock_positions.shape == path.shape
    assert result.cash_account.shape == path.shape
    assert result.transaction_costs.shape == path.shape
    assert np.isfinite(result.final_pnl)
    assert result.terminal_payoff == 5.0


def test_more_frequent_rebalancing_reduces_average_hedge_error() -> None:
    simulator = HedgingSimulator(option=_option(), market=_market())
    _, coarse = simulator.simulate_delta_hedging(n_paths=40, steps=6, seed=11)
    _, fine = simulator.simulate_delta_hedging(n_paths=40, steps=36, seed=11)

    assert fine.mean_absolute_error < coarse.mean_absolute_error


def test_transaction_costs_reduce_short_option_hedge_pnl() -> None:
    path = np.array([100.0, 102.0, 98.0, 104.0, 101.0])
    no_cost = HedgingSimulator(
        option=_option(),
        market=_market(),
        transaction_cost_rate=0.0,
    ).simulate_delta_hedge_path(path)
    with_cost = HedgingSimulator(
        option=_option(),
        market=_market(),
        transaction_cost_rate=0.001,
    ).simulate_delta_hedge_path(path)

    assert np.sum(with_cost.transaction_costs) > 0.0
    assert with_cost.final_pnl < no_cost.final_pnl


def test_gamma_vega_neutral_option_positions() -> None:
    target = HedgePortfolioLeg(
        quantity=1.0,
        greeks=Greeks(delta=0.5, gamma=10.0, vega=40.0, theta=-5.0, rho=8.0),
    )
    gamma_hedge = Greeks(delta=0.4, gamma=5.0, vega=10.0, theta=-3.0, rho=4.0)
    vega_hedge = Greeks(delta=-0.2, gamma=2.0, vega=20.0, theta=-2.0, rho=-1.0)

    q_gamma, q_vega = gamma_vega_neutral_option_positions(
        target=target,
        gamma_hedge=gamma_hedge,
        vega_hedge=vega_hedge,
    )

    total_gamma = target.greeks.gamma + q_gamma * gamma_hedge.gamma + q_vega * vega_hedge.gamma
    total_vega = target.greeks.vega + q_gamma * gamma_hedge.vega + q_vega * vega_hedge.vega

    assert total_gamma == pytest.approx(0.0)
    assert total_vega == pytest.approx(0.0)


def test_gamma_vega_neutral_rejects_singular_hedges() -> None:
    target = HedgePortfolioLeg(
        quantity=1.0,
        greeks=Greeks(delta=0.5, gamma=10.0, vega=40.0, theta=-5.0, rho=8.0),
    )
    first = Greeks(delta=0.0, gamma=1.0, vega=2.0, theta=0.0, rho=0.0)
    second = Greeks(delta=0.0, gamma=2.0, vega=4.0, theta=0.0, rho=0.0)

    with pytest.raises(ValueError, match="cannot span"):
        gamma_vega_neutral_option_positions(target=target, gamma_hedge=first, vega_hedge=second)


def test_hedging_simulator_validates_inputs() -> None:
    with pytest.raises(ValueError, match="option_quantity must be non-zero"):
        HedgingSimulator(option=_option(), market=_market(), option_quantity=0.0)

    with pytest.raises(ValueError, match="transaction_cost_rate must be non-negative"):
        HedgingSimulator(option=_option(), market=_market(), transaction_cost_rate=-0.01)


def test_hedging_plots_return_figures(tmp_path) -> None:
    simulator = HedgingSimulator(option=_option(), market=_market())
    result = simulator.simulate_delta_hedge_path(np.array([100.0, 102.0, 101.0, 105.0]))

    pnl_fig = plot_hedging_pnl(result, output_path=str(tmp_path / "hedge_pnl.png"))
    dist_fig = plot_hedging_error_distribution(
        np.array([result.final_pnl, -result.final_pnl]),
        output_path=str(tmp_path / "hedge_errors.png"),
    )

    assert len(pnl_fig.axes) == 3
    assert len(dist_fig.axes) == 1
