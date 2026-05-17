"""Volatility and risk-regime features."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_volatility_features(
    frame: pd.DataFrame,
    *,
    volatility_windows: tuple[int, ...],
    atr_window: int,
) -> pd.DataFrame:
    """Create rolling volatility, ATR, and volatility-regime features."""

    features = pd.DataFrame(index=frame.index)
    close = frame["close"].astype(float)
    log_return = np.log(close).diff()

    for window in volatility_windows:
        volatility = log_return.rolling(window).std(ddof=1) * np.sqrt(252)
        features[f"rolling_volatility_{window}"] = volatility
        features[f"volatility_regime_{window}"] = classify_volatility_regime(volatility, window)

    features[f"atr_{atr_window}"] = average_true_range(frame, window=atr_window)
    features[f"atr_ratio_{atr_window}"] = features[f"atr_{atr_window}"] / close

    return features


def average_true_range(frame: pd.DataFrame, window: int = 14) -> pd.Series:
    """Compute average true range from OHLC data."""

    high = frame["high"].astype(float)
    low = frame["low"].astype(float)
    close = frame["close"].astype(float)
    previous_close = close.shift(1)

    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(window).mean()


def classify_volatility_regime(volatility: pd.Series, window: int) -> pd.Series:
    """Classify volatility as low, normal, or high using trailing quantiles.

    The thresholds are shifted by one row so today's regime does not use today's
    realized volatility to define its own bucket.
    """

    rolling = volatility.rolling(window, min_periods=max(5, window // 2))
    low_threshold = rolling.quantile(0.33).shift(1)
    high_threshold = rolling.quantile(0.66).shift(1)

    regime = pd.Series(0.0, index=volatility.index)
    regime = regime.mask(volatility <= low_threshold, -1.0)
    regime = regime.mask(volatility >= high_threshold, 1.0)
    return regime
