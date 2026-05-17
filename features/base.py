"""Base contracts for factor and technical-feature pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import pandas as pd


@dataclass(frozen=True)
class FeatureConfig:
    """Common rolling windows used by the feature engine."""

    return_windows: tuple[int, ...] = (1, 5, 10, 20)
    volatility_windows: tuple[int, ...] = (10, 20, 60)
    momentum_windows: tuple[int, ...] = (10, 20, 50)
    rsi_window: int = 14
    ema_fast: int = 12
    ema_slow: int = 26
    macd_signal: int = 9
    atr_window: int = 14
    volume_window: int = 20
    structure_window: int = 20
    fractal_order: int = 2
    drop_na: bool = True


class FeatureEngineer(Protocol):
    """Protocol implemented by all feature engineering components."""

    def transform(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Create feature columns from a clean OHLCV price frame."""


def validate_feature_frame(frame: pd.DataFrame) -> None:
    """Validate that a feature matrix is index-aligned and finite enough for modeling."""

    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError("feature frame must use a DatetimeIndex")
    if frame.empty:
        raise ValueError("feature frame is empty")
