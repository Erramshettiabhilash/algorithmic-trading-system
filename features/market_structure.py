"""Market-structure features for price-action research."""

from __future__ import annotations

import pandas as pd


def add_market_structure_features(
    frame: pd.DataFrame,
    *,
    structure_window: int,
    fractal_order: int,
) -> pd.DataFrame:
    """Create confirmed fractals, liquidity sweeps, and trend-structure features."""

    features = pd.DataFrame(index=frame.index)
    high = frame["high"].astype(float)
    low = frame["low"].astype(float)
    close = frame["close"].astype(float)

    features[f"confirmed_fractal_high_{fractal_order}"] = confirmed_fractal_high(
        high,
        order=fractal_order,
    )
    features[f"confirmed_fractal_low_{fractal_order}"] = confirmed_fractal_low(
        low,
        order=fractal_order,
    )

    prior_high = high.shift(1).rolling(structure_window).max()
    prior_low = low.shift(1).rolling(structure_window).min()
    features[f"liquidity_sweep_high_{structure_window}"] = (
        (high > prior_high) & (close < prior_high)
    ).astype(float)
    features[f"liquidity_sweep_low_{structure_window}"] = (
        (low < prior_low) & (close > prior_low)
    ).astype(float)

    features[f"trend_position_{structure_window}"] = (close - prior_low) / (
        prior_high - prior_low
    ).replace(0.0, pd.NA)
    features[f"higher_high_{structure_window}"] = (high > prior_high).astype(float)
    features[f"lower_low_{structure_window}"] = (low < prior_low).astype(float)
    features[f"trend_structure_{structure_window}"] = (
        features[f"higher_high_{structure_window}"]
        - features[f"lower_low_{structure_window}"]
    )

    return features


def confirmed_fractal_high(high: pd.Series, order: int = 2) -> pd.Series:
    """Return confirmed fractal highs aligned to the confirmation timestamp."""

    window = 2 * order + 1
    centered_max = high.rolling(window, center=True).max()
    raw_fractal = (high == centered_max).astype(float)
    return raw_fractal.shift(order)


def confirmed_fractal_low(low: pd.Series, order: int = 2) -> pd.Series:
    """Return confirmed fractal lows aligned to the confirmation timestamp."""

    window = 2 * order + 1
    centered_min = low.rolling(window, center=True).min()
    raw_fractal = (low == centered_min).astype(float)
    return raw_fractal.shift(order)
