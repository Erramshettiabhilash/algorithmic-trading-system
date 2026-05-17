"""Smoke tests for the initial project scaffold."""

from __future__ import annotations

import pytest

from utils.instruments import EuropeanOption, OptionType
from utils.market import MarketEnvironment


def test_can_create_european_option() -> None:
    option = EuropeanOption(strike=100.0, maturity=1.0, option_type=OptionType.CALL)

    assert option.strike == 100.0
    assert option.maturity == 1.0
    assert option.option_type is OptionType.CALL


def test_can_create_market_environment() -> None:
    market = MarketEnvironment(spot=100.0, risk_free_rate=0.05, volatility=0.2)

    assert market.spot == 100.0
    assert market.risk_free_rate == 0.05
    assert market.volatility == 0.2


def test_option_requires_positive_strike() -> None:
    with pytest.raises(ValueError, match="strike must be positive"):
        EuropeanOption(strike=0.0, maturity=1.0, option_type=OptionType.CALL)


def test_market_requires_positive_spot() -> None:
    with pytest.raises(ValueError, match="spot must be positive"):
        MarketEnvironment(spot=-1.0, risk_free_rate=0.05, volatility=0.2)


def test_market_allows_negative_rates() -> None:
    market = MarketEnvironment(spot=100.0, risk_free_rate=-0.01, volatility=0.2)

    assert market.risk_free_rate == -0.01
