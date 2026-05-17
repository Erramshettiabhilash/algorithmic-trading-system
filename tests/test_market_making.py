"""Tests for the Step 10 options market-making engine."""

from __future__ import annotations

import pytest

from analytics.market_making import (
    InventoryState,
    MarketMakingConfig,
    MarketMakingEngine,
)
from models.black_scholes import BlackScholesModel
from utils.instruments import EuropeanOption, OptionType
from utils.market import MarketEnvironment


def _option() -> EuropeanOption:
    return EuropeanOption(strike=100.0, maturity=1.0, option_type=OptionType.CALL)


def _market() -> MarketEnvironment:
    return MarketEnvironment(spot=100.0, risk_free_rate=0.05, volatility=0.2)


def _engine() -> MarketMakingEngine:
    config = MarketMakingConfig(
        base_spread=0.10,
        min_spread=0.01,
        inventory_limit=20.0,
        base_quote_size=10.0,
        inventory_spread_sensitivity=0.50,
        inventory_skew_sensitivity=0.50,
        gamma_spread_sensitivity=1.00,
        vega_spread_sensitivity=0.001,
        adverse_selection_buffer=0.02,
    )
    return MarketMakingEngine(option=_option(), market=_market(), config=config)


def test_fair_value_matches_black_scholes() -> None:
    engine = _engine()

    assert engine.fair_value() == pytest.approx(
        BlackScholesModel(option=_option(), market=_market()).price()
    )


def test_flat_inventory_quote_surrounds_fair_value() -> None:
    quote = _engine().quote(InventoryState(option_position=0.0))

    assert quote.bid < quote.fair_value < quote.ask
    assert quote.spread >= _engine().config.min_spread
    assert quote.quote_size == pytest.approx(_engine().config.base_quote_size)
    assert quote.inventory_skew == pytest.approx(0.0)


def test_long_inventory_skews_quote_lower_and_widens_spread() -> None:
    engine = _engine()
    flat = engine.quote(InventoryState(option_position=0.0))
    long_inventory = engine.quote(InventoryState(option_position=15.0))

    assert long_inventory.mid < flat.mid
    assert long_inventory.bid < flat.bid
    assert long_inventory.spread > flat.spread
    assert long_inventory.quote_size < flat.quote_size
    assert long_inventory.gamma_exposure > 0.0
    assert long_inventory.vega_exposure > 0.0
    assert long_inventory.theta_decay < 0.0


def test_short_inventory_skews_quote_higher() -> None:
    engine = _engine()
    flat = engine.quote(InventoryState(option_position=0.0))
    short_inventory = engine.quote(InventoryState(option_position=-15.0))

    assert short_inventory.mid > flat.mid
    assert short_inventory.ask > flat.ask
    assert short_inventory.spread > flat.spread
    assert short_inventory.gamma_exposure < 0.0


def test_execute_trade_buy_increases_inventory_and_reduces_cash() -> None:
    engine = _engine()
    inventory = InventoryState()
    quote = engine.quote(inventory)

    updated = engine.execute_trade(inventory=inventory, side="buy", quantity=2.0, quote=quote)

    assert updated.option_position == 2.0
    assert updated.cash == pytest.approx(-2.0 * quote.bid)


def test_execute_trade_sell_decreases_inventory_and_increases_cash() -> None:
    engine = _engine()
    inventory = InventoryState()
    quote = engine.quote(inventory)

    updated = engine.execute_trade(inventory=inventory, side="sell", quantity=3.0, quote=quote)

    assert updated.option_position == -3.0
    assert updated.cash == pytest.approx(3.0 * quote.ask)


def test_execute_trade_rejects_invalid_side() -> None:
    with pytest.raises(ValueError, match="side must be"):
        _engine().execute_trade(inventory=InventoryState(), side="hold", quantity=1.0)


def test_inventory_warning_triggers_at_limit() -> None:
    engine = _engine()

    assert not engine.inventory_warning(InventoryState(option_position=19.0))
    assert engine.inventory_warning(InventoryState(option_position=20.0))
    assert engine.inventory_warning(InventoryState(option_position=-25.0))


def test_market_making_config_validates_inputs() -> None:
    with pytest.raises(ValueError, match="inventory_limit must be positive"):
        MarketMakingConfig(inventory_limit=0.0)

    with pytest.raises(ValueError, match="base_spread must be non-negative"):
        MarketMakingConfig(base_spread=-0.01)
