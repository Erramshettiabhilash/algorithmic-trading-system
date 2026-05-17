"""Tests for the Step 4 Black-Scholes Greeks engine."""

from __future__ import annotations

from dataclasses import replace

import pytest

from analytics.greeks import GreeksEngine, standard_normal_pdf
from models.black_scholes import BlackScholesModel
from utils.instruments import EuropeanOption, OptionType
from utils.market import MarketEnvironment


def _model(
    option_type: OptionType,
    spot: float = 100.0,
    volatility: float = 0.2,
) -> BlackScholesModel:
    option = EuropeanOption(strike=100.0, maturity=1.0, option_type=option_type)
    market = MarketEnvironment(spot=spot, risk_free_rate=0.05, volatility=volatility)
    return BlackScholesModel(option=option, market=market)


def test_standard_normal_pdf_known_value() -> None:
    assert standard_normal_pdf(0.0) == pytest.approx(0.3989422804)


def test_call_greeks_known_values() -> None:
    greeks = GreeksEngine(model=_model(OptionType.CALL)).all()

    assert greeks.delta == pytest.approx(0.6368, abs=1e-4)
    assert greeks.gamma == pytest.approx(0.0188, abs=1e-4)
    assert greeks.vega == pytest.approx(37.5240, abs=1e-4)
    assert greeks.theta == pytest.approx(-6.4140, abs=1e-4)
    assert greeks.rho == pytest.approx(53.2325, abs=1e-4)
    assert greeks.vega_per_vol_point == pytest.approx(0.3752, abs=1e-4)
    assert greeks.rho_per_rate_point == pytest.approx(0.5323, abs=1e-4)


def test_put_greeks_known_values() -> None:
    greeks = GreeksEngine(model=_model(OptionType.PUT)).all()

    assert greeks.delta == pytest.approx(-0.3632, abs=1e-4)
    assert greeks.gamma == pytest.approx(0.0188, abs=1e-4)
    assert greeks.vega == pytest.approx(37.5240, abs=1e-4)
    assert greeks.theta == pytest.approx(-1.6579, abs=1e-4)
    assert greeks.rho == pytest.approx(-41.8905, abs=1e-4)


def test_call_delta_matches_finite_difference() -> None:
    model = _model(OptionType.CALL)
    bump = 0.01
    up_market = replace(model.market, spot=model.market.spot + bump)
    down_market = replace(model.market, spot=model.market.spot - bump)
    up_price = BlackScholesModel(option=model.option, market=up_market).price()
    down_price = BlackScholesModel(option=model.option, market=down_market).price()
    finite_difference_delta = (up_price - down_price) / (2.0 * bump)

    assert GreeksEngine(model=model).delta() == pytest.approx(finite_difference_delta, abs=1e-5)


def test_vega_matches_finite_difference() -> None:
    model = _model(OptionType.CALL)
    bump = 0.0001
    up_market = replace(model.market, volatility=model.market.volatility + bump)
    down_market = replace(model.market, volatility=model.market.volatility - bump)
    up_price = BlackScholesModel(option=model.option, market=up_market).price()
    down_price = BlackScholesModel(option=model.option, market=down_market).price()
    finite_difference_vega = (up_price - down_price) / (2.0 * bump)

    assert GreeksEngine(model=model).vega() == pytest.approx(finite_difference_vega, abs=1e-5)


def test_theta_per_day_uses_trading_days() -> None:
    engine = GreeksEngine(model=_model(OptionType.CALL))

    assert engine.theta_per_day(trading_days=252) == pytest.approx(engine.theta() / 252)


def test_theta_per_day_rejects_invalid_trading_days() -> None:
    engine = GreeksEngine(model=_model(OptionType.CALL))

    with pytest.raises(ValueError, match="trading_days must be positive"):
        engine.theta_per_day(trading_days=0)


def test_greeks_reject_zero_volatility() -> None:
    engine = GreeksEngine(model=_model(OptionType.CALL, volatility=0.0))

    with pytest.raises(ValueError, match="undefined"):
        engine.delta()
