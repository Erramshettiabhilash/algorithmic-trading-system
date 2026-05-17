"""Tests for the Step 5 Monte Carlo engine."""

from __future__ import annotations

import numpy as np
import pytest

from models.black_scholes import BlackScholesModel
from simulations.monte_carlo import MonteCarloEngine
from utils.instruments import EuropeanOption, OptionType
from utils.market import MarketEnvironment


def _option(option_type: OptionType = OptionType.CALL) -> EuropeanOption:
    return EuropeanOption(strike=100.0, maturity=1.0, option_type=option_type)


def _market() -> MarketEnvironment:
    return MarketEnvironment(spot=100.0, risk_free_rate=0.05, volatility=0.2)


def test_simulated_paths_shape_and_initial_spot() -> None:
    engine = MonteCarloEngine(
        option=_option(),
        market=_market(),
        n_paths=100,
        steps=12,
        seed=42,
    )

    paths = engine.simulate_paths()

    assert paths.shape == (100, 13)
    np.testing.assert_allclose(paths[:, 0], np.full(100, 100.0))
    assert np.all(paths > 0.0)


def test_call_payoff_from_terminal_prices() -> None:
    engine = MonteCarloEngine(option=_option(OptionType.CALL), market=_market())
    terminal_prices = np.array([80.0, 100.0, 120.0])

    np.testing.assert_allclose(engine.payoff(terminal_prices), np.array([0.0, 0.0, 20.0]))


def test_put_payoff_from_terminal_prices() -> None:
    engine = MonteCarloEngine(option=_option(OptionType.PUT), market=_market())
    terminal_prices = np.array([80.0, 100.0, 120.0])

    np.testing.assert_allclose(engine.payoff(terminal_prices), np.array([20.0, 0.0, 0.0]))


def test_monte_carlo_price_is_reproducible_with_seed() -> None:
    first = MonteCarloEngine(
        option=_option(),
        market=_market(),
        n_paths=5_000,
        steps=50,
        seed=7,
    ).price()
    second = MonteCarloEngine(
        option=_option(),
        market=_market(),
        n_paths=5_000,
        steps=50,
        seed=7,
    ).price()

    assert first.price == second.price
    assert first.standard_error == second.standard_error
    assert first.confidence_interval == second.confidence_interval


def test_monte_carlo_call_price_converges_toward_black_scholes() -> None:
    option = _option(OptionType.CALL)
    market = _market()
    analytical_price = BlackScholesModel(option=option, market=market).price()
    result = MonteCarloEngine(
        option=option,
        market=market,
        n_paths=80_000,
        steps=80,
        seed=11,
    ).price()

    assert result.price == pytest.approx(analytical_price, abs=0.12)
    assert result.confidence_interval[0] < analytical_price < result.confidence_interval[1]


def test_monte_carlo_put_price_converges_toward_black_scholes() -> None:
    option = _option(OptionType.PUT)
    market = _market()
    analytical_price = BlackScholesModel(option=option, market=market).price()
    result = MonteCarloEngine(
        option=option,
        market=market,
        n_paths=80_000,
        steps=80,
        seed=11,
    ).price()

    assert result.price == pytest.approx(analytical_price, abs=0.12)
    assert result.confidence_interval[0] < analytical_price < result.confidence_interval[1]


def test_monte_carlo_rejects_invalid_path_count() -> None:
    with pytest.raises(ValueError, match="n_paths must be greater than 1"):
        MonteCarloEngine(option=_option(), market=_market(), n_paths=1)


def test_monte_carlo_rejects_invalid_step_count() -> None:
    with pytest.raises(ValueError, match="steps must be positive"):
        MonteCarloEngine(option=_option(), market=_market(), steps=0)

