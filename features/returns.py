"""Return-based features for predictive financial modeling."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_return_features(frame: pd.DataFrame, windows: tuple[int, ...]) -> pd.DataFrame:
    """Create log-return, rolling-return, and autocorrelation features."""

    features = pd.DataFrame(index=frame.index)
    close = frame["close"].astype(float)
    log_return = np.log(close).diff()

    features["log_return_1"] = log_return
    features["simple_return_1"] = close.pct_change()

    for window in windows:
        features[f"log_return_{window}"] = np.log(close / close.shift(window))
        features[f"rolling_return_{window}"] = close.pct_change(window)
        features[f"return_autocorr_{window}"] = log_return.rolling(window).apply(
            _safe_autocorrelation,
            raw=False,
        )

    return features


def _safe_autocorrelation(values: pd.Series) -> float:
    """Return lag-1 autocorrelation while tolerating flat windows."""

    autocorrelation = values.autocorr(lag=1)
    if pd.isna(autocorrelation):
        return 0.0
    return float(autocorrelation)
