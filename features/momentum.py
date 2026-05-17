"""Momentum and trend-following technical features."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_momentum_features(
    frame: pd.DataFrame,
    *,
    momentum_windows: tuple[int, ...],
    rsi_window: int,
    ema_fast: int,
    ema_slow: int,
    macd_signal: int,
) -> pd.DataFrame:
    """Create RSI, EMA crossover, MACD, and momentum-factor features."""

    features = pd.DataFrame(index=frame.index)
    close = frame["close"].astype(float)

    features[f"rsi_{rsi_window}"] = relative_strength_index(close, window=rsi_window)

    fast = close.ewm(span=ema_fast, adjust=False, min_periods=ema_fast).mean()
    slow = close.ewm(span=ema_slow, adjust=False, min_periods=ema_slow).mean()
    features[f"ema_{ema_fast}"] = fast
    features[f"ema_{ema_slow}"] = slow
    features[f"ema_crossover_{ema_fast}_{ema_slow}"] = fast / slow - 1.0

    macd_line = fast - slow
    signal = macd_line.ewm(span=macd_signal, adjust=False, min_periods=macd_signal).mean()
    features["macd_line"] = macd_line
    features["macd_signal"] = signal
    features["macd_histogram"] = macd_line - signal

    for window in momentum_windows:
        features[f"momentum_factor_{window}"] = close / close.shift(window) - 1.0

    return features


def relative_strength_index(close: pd.Series, window: int = 14) -> pd.Series:
    """Compute RSI using exponentially smoothed gains and losses."""

    delta = close.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)
    average_gain = gains.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    average_loss = losses.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    relative_strength = average_gain / average_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + relative_strength))
    rsi = rsi.mask((average_loss == 0.0) & (average_gain > 0.0), 100.0)
    rsi = rsi.mask((average_loss == 0.0) & (average_gain == 0.0), 50.0)
    return rsi
