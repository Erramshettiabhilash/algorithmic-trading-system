"""Tests for the Step 3 Black-Scholes implementation."""

from __future__ import annotations

import pytest

from models.black_scholes import BlackScholesModel, standard_normal_cdf
from utils.instruments import EuropeanOption, OptionType
from utils.market import MarketEnvironment


def test_standard_normal_cdf_known_values() -> None:
    assert standard_normal_cdf(0.0) == pytest.approx(0.5)
    assert standard_normal_cdf(1.96) == pytest.approx(0.975, abs=5e-4)
    assert standard_normal_cdf(-1.96) == pytest.approx(0.025, abs=5e-4)


def test_black_scholes_call_price_known_value() -> None:
    option = EuropeanOption(strike=100.0, maturity=1.0, option_type=OptionType.CALL)
    market = MarketEnvironment(spot=100.0, risk_free_rate=0.05, volatility=0.2)
    model = BlackScholesModel(option=option, market=market)

    assert model.d1 == pytest.approx(0.35)
    assert model.d2 == pytest.approx(0.15)
    assert model.price() == pytest.approx(10.4506, abs=1e-4)


def test_black_scholes_put_price_known_value() -> None:
    option = EuropeanOption(strike=100.0, maturity=1.0, option_type=OptionType.PUT)
    market = MarketEnvironment(spot=100.0, risk_free_rate=0.05, volatility=0.2)
    model = BlackScholesModel(option=option, market=market)

    assert model.price() == pytest.approx(5.5735, abs=1e-4)


def test_put_call_parity() -> None:
    call = EuropeanOption(strike=100.0, maturity=1.0, option_type=OptionType.CALL)
    put = EuropeanOption(strike=100.0, maturity=1.0, option_type=OptionType.PUT)
    market = MarketEnvironment(spot=100.0, risk_free_rate=0.05, volatility=0.2)

    call_price = BlackScholesModel(option=call, market=market).price()
    put_price = BlackScholesModel(option=put, market=market).price()
    discounted_strike = call.strike * BlackScholesModel(option=call, market=market).discount_factor

    assert call_price - put_price == pytest.approx(market.spot - discounted_strike)


def test_zero_volatility_call_uses_discounted_deterministic_payoff() -> None:
    option = EuropeanOption(strike=100.0, maturity=1.0, option_type=OptionType.CALL)
    market = MarketEnvironment(spot=100.0, risk_free_rate=0.05, volatility=0.0)

    assert BlackScholesModel(option=option, market=market).price() == pytest.approx(
        4.8771,
        abs=1e-4,
    )


def test_d1_rejects_zero_volatility() -> None:
    option = EuropeanOption(strike=100.0, maturity=1.0, option_type=OptionType.CALL)
    market = MarketEnvironment(spot=100.0, risk_free_rate=0.05, volatility=0.0)
    model = BlackScholesModel(option=option, market=market)

    with pytest.raises(ValueError, match="undefined"):
        _ = model.d1
