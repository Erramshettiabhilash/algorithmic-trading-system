"""Tests for the Step 11 stress-testing framework."""

from __future__ import annotations

import numpy as np
import pytest

from analytics.stress_testing import (
    OptionPosition,
    StressScenario,
    StressTester,
    standard_stress_scenarios,
)
from utils.instruments import EuropeanOption, OptionType
from utils.market import MarketEnvironment
from visualization.charts import plot_stress_pnl_distribution, plot_stress_scenario_pnl


def _market() -> MarketEnvironment:
    return MarketEnvironment(spot=100.0, risk_free_rate=0.03, volatility=0.2)


def _call() -> EuropeanOption:
    return EuropeanOption(strike=100.0, maturity=1.0, option_type=OptionType.CALL)


def _put() -> EuropeanOption:
    return EuropeanOption(strike=95.0, maturity=1.0, option_type=OptionType.PUT)


def test_value_portfolio_aggregates_option_values_and_greeks() -> None:
    tester = StressTester(
        positions=(OptionPosition(_call(), 1.0), OptionPosition(_put(), -2.0)),
        market=_market(),
    )

    valuation = tester.value_portfolio()

    assert np.isfinite(valuation.value)
    assert np.isfinite(valuation.greeks.delta)
    assert np.isfinite(valuation.greeks.gamma)
    assert np.isfinite(valuation.greeks.vega)


def test_volatility_explosion_benefits_long_option_portfolio() -> None:
    tester = StressTester(positions=(OptionPosition(_call(), 1.0),), market=_market())
    result = tester.run_scenario(StressScenario(name="Vol up", volatility_multiplier=2.0))

    assert result.stressed_market.volatility == pytest.approx(0.4)
    assert result.pnl > 0.0
    assert result.stressed_greeks.vega > 0.0


def test_market_crash_hurts_short_put_position() -> None:
    tester = StressTester(positions=(OptionPosition(_put(), -1.0),), market=_market())
    result = tester.run_scenario(
        StressScenario(name="Crash", spot_shock=-0.25, volatility_multiplier=1.5)
    )

    assert result.stressed_market.spot == pytest.approx(75.0)
    assert result.pnl < 0.0


def test_run_scenarios_requires_non_empty_list() -> None:
    tester = StressTester(positions=(OptionPosition(_call(), 1.0),), market=_market())

    with pytest.raises(ValueError, match="at least one"):
        tester.run_scenarios([])


def test_standard_stress_scenarios_include_named_shocks() -> None:
    names = {scenario.name for scenario in standard_stress_scenarios()}

    assert "Volatility explosion" in names
    assert "Market crash" in names
    assert "Flash crash" in names
    assert "Term-structure inversion proxy" in names


def test_scenario_rejects_invalid_spot_crash() -> None:
    with pytest.raises(ValueError, match="greater than -100%"):
        StressScenario(name="Impossible", spot_shock=-1.0)


def test_scenario_rejects_negative_stressed_volatility() -> None:
    scenario = StressScenario(name="Negative vol", volatility_shock=-0.25)

    with pytest.raises(ValueError, match="negative volatility"):
        scenario.apply(_market())


def test_monte_carlo_stress_returns_tail_metrics() -> None:
    tester = StressTester(positions=(OptionPosition(_put(), -1.0),), market=_market())
    result = tester.monte_carlo_stress(
        n_paths=2_000,
        horizon=5.0 / 252.0,
        realized_volatility=0.45,
        volatility_multiplier=1.5,
        spot_jump=-0.03,
        seed=7,
    )

    assert result.pnl.shape == (2_000,)
    assert result.n_paths == 2_000
    assert result.std_pnl > 0.0
    assert result.value_at_risk <= result.mean_pnl
    assert result.expected_shortfall <= result.value_at_risk


def test_monte_carlo_stress_reduces_maturity_after_horizon() -> None:
    tester = StressTester(positions=(OptionPosition(_call(), 1.0),), market=_market())

    short_horizon = tester.monte_carlo_stress(n_paths=500, horizon=1.0 / 252.0, seed=1)
    long_horizon = tester.monte_carlo_stress(n_paths=500, horizon=0.5, seed=1)

    assert np.isfinite(short_horizon.mean_pnl)
    assert np.isfinite(long_horizon.mean_pnl)


def test_stress_plots_return_figures(tmp_path) -> None:
    tester = StressTester(positions=(OptionPosition(_call(), 1.0),), market=_market())
    scenario_results = tester.run_scenarios(standard_stress_scenarios()[:2])
    mc_result = tester.monte_carlo_stress(n_paths=200, seed=3)

    scenario_fig = plot_stress_scenario_pnl(
        scenario_results,
        output_path=str(tmp_path / "scenario_pnl.png"),
    )
    distribution_fig = plot_stress_pnl_distribution(
        mc_result.pnl,
        value_at_risk=mc_result.value_at_risk,
        expected_shortfall=mc_result.expected_shortfall,
        output_path=str(tmp_path / "stress_distribution.png"),
    )

    assert len(scenario_fig.axes) == 1
    assert len(distribution_fig.axes) == 1

