"""Tests for the Step 6 implied-volatility engine."""

from __future__ import annotations

import pandas as pd
import pytest

from analytics.implied_volatility import (
    ImpliedVolatilitySolver,
    extract_implied_volatilities,
)
from models.black_scholes import BlackScholesModel
from utils.instruments import EuropeanOption, OptionType
from utils.market import MarketEnvironment


def _market(volatility: float = 0.2) -> MarketEnvironment:
    return MarketEnvironment(spot=100.0, risk_free_rate=0.05, volatility=volatility)


def _option(option_type: OptionType = OptionType.CALL, strike: float = 100.0) -> EuropeanOption:
    return EuropeanOption(strike=strike, maturity=1.0, option_type=option_type)


def _price_from_volatility(volatility: float, option_type: OptionType = OptionType.CALL) -> float:
    option = _option(option_type=option_type)
    market = _market(volatility=volatility)
    return BlackScholesModel(option=option, market=market).price()


def test_bisection_recovers_call_implied_volatility() -> None:
    option = _option(OptionType.CALL)
    market_price = _price_from_volatility(0.35, OptionType.CALL)
    solver = ImpliedVolatilitySolver(option=option, market=_market(), market_price=market_price)

    result = solver.solve_bisection()

    assert result.converged
    assert result.method == "bisection"
    assert result.implied_volatility == pytest.approx(0.35, abs=1e-7)
    assert abs(result.pricing_error) <= solver.tolerance


def test_newton_recovers_put_implied_volatility() -> None:
    option = _option(OptionType.PUT)
    market_price = _price_from_volatility(0.28, OptionType.PUT)
    solver = ImpliedVolatilitySolver(option=option, market=_market(), market_price=market_price)

    result = solver.solve_newton(initial_volatility=0.2)

    assert result.converged
    assert result.method == "newton"
    assert result.implied_volatility == pytest.approx(0.28, abs=1e-7)


def test_newton_and_bisection_agree() -> None:
    option = _option(OptionType.CALL)
    market_price = _price_from_volatility(0.42, OptionType.CALL)
    solver = ImpliedVolatilitySolver(option=option, market=_market(), market_price=market_price)

    newton = solver.solve_newton(initial_volatility=0.2)
    bisection = solver.solve_bisection()

    assert newton.implied_volatility == pytest.approx(bisection.implied_volatility, abs=1e-7)


def test_solver_rejects_price_outside_no_arbitrage_bounds() -> None:
    option = _option(OptionType.CALL)

    with pytest.raises(ValueError, match="no-arbitrage bounds"):
        ImpliedVolatilitySolver(option=option, market=_market(), market_price=150.0)


def test_bisection_rejects_unbracketed_range() -> None:
    option = _option(OptionType.CALL)
    market_price = _price_from_volatility(0.35, OptionType.CALL)
    solver = ImpliedVolatilitySolver(option=option, market=_market(), market_price=market_price)

    with pytest.raises(ValueError, match="not bracketed"):
        solver.solve_bisection(min_volatility=0.01, max_volatility=0.02)


def test_extract_implied_volatilities_from_option_chain() -> None:
    spot = 100.0
    risk_free_rate = 0.05
    strikes = [90.0, 100.0, 110.0]
    true_vols = [0.30, 0.24, 0.28]
    prices = []

    for strike, volatility in zip(strikes, true_vols, strict=True):
        option = _option(option_type=OptionType.CALL, strike=strike)
        market = MarketEnvironment(
            spot=spot,
            risk_free_rate=risk_free_rate,
            volatility=volatility,
        )
        prices.append(BlackScholesModel(option=option, market=market).price())

    option_chain = pd.DataFrame(
        {
            "strike": strikes,
            "maturity": [1.0, 1.0, 1.0],
            "option_type": ["call", "call", "call"],
            "market_price": prices,
        }
    )

    extracted = extract_implied_volatilities(
        option_chain=option_chain,
        spot=spot,
        risk_free_rate=risk_free_rate,
        method="bisection",
    )

    assert list(extracted["moneyness"]) == pytest.approx([0.9, 1.0, 1.1])
    assert list(extracted["implied_volatility"]) == pytest.approx(true_vols, abs=1e-7)
    assert extracted["iv_converged"].all()


def test_extract_implied_volatilities_requires_columns() -> None:
    option_chain = pd.DataFrame({"strike": [100.0]})

    with pytest.raises(ValueError, match="missing required columns"):
        extract_implied_volatilities(
            option_chain=option_chain,
            spot=100.0,
            risk_free_rate=0.05,
        )

