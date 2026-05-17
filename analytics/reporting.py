"""Reusable analytics summaries for options research reports."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from analytics.greeks import Greeks, GreeksEngine
from models.black_scholes import BlackScholesModel
from utils.instruments import EuropeanOption
from utils.market import MarketEnvironment


@dataclass(frozen=True)
class OptionAnalyticsSnapshot:
    """One-option pricing and Greeks summary."""

    option: EuropeanOption
    market: MarketEnvironment
    price: float
    greeks: Greeks


def option_analytics_snapshot(
    option: EuropeanOption,
    market: MarketEnvironment,
) -> OptionAnalyticsSnapshot:
    """Return price and Greeks for one option."""
    model = BlackScholesModel(option=option, market=market)
    greeks = GreeksEngine(model=model).all()
    return OptionAnalyticsSnapshot(
        option=option,
        market=market,
        price=model.price(),
        greeks=greeks,
    )


def option_chain_analytics(
    options: list[EuropeanOption],
    market: MarketEnvironment,
) -> pd.DataFrame:
    """Return a tidy analytics table for a list of European options."""
    if not options:
        raise ValueError("options must contain at least one option.")

    rows = []
    for option in options:
        snapshot = option_analytics_snapshot(option=option, market=market)
        rows.append(
            {
                "option_type": option.option_type.value,
                "strike": option.strike,
                "maturity": option.maturity,
                "spot": market.spot,
                "risk_free_rate": market.risk_free_rate,
                "volatility": market.volatility,
                "price": snapshot.price,
                "delta": snapshot.greeks.delta,
                "gamma": snapshot.greeks.gamma,
                "vega": snapshot.greeks.vega,
                "theta": snapshot.greeks.theta,
                "rho": snapshot.greeks.rho,
            }
        )

    return pd.DataFrame(rows).sort_values(["maturity", "strike", "option_type"]).reset_index(
        drop=True
    )

