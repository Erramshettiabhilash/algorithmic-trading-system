"""Volume-flow features."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_volume_features(frame: pd.DataFrame, *, window: int) -> pd.DataFrame:
    """Create OBV, volume ratio, and volume z-score features."""

    features = pd.DataFrame(index=frame.index)
    close = frame["close"].astype(float)
    volume = frame["volume"].astype(float)

    features["obv"] = on_balance_volume(close, volume)
    features[f"volume_ratio_{window}"] = volume / volume.rolling(window).mean()

    rolling_mean = volume.rolling(window).mean()
    rolling_std = volume.rolling(window).std(ddof=0)
    features[f"volume_zscore_{window}"] = (volume - rolling_mean) / rolling_std.replace(0.0, np.nan)

    return features


def on_balance_volume(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Compute On-Balance Volume as cumulative signed volume."""

    direction = np.sign(close.diff()).fillna(0.0)
    return (direction * volume).cumsum()
