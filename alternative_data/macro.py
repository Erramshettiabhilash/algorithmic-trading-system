"""Macroeconomic indicator feature engineering."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class MacroFeatureConfig:
    """Configuration for macro feature transformations."""

    zscore_window: int = 12
    change_periods: int = 1


def create_macro_features(
    macro_data: pd.DataFrame,
    config: MacroFeatureConfig | None = None,
) -> pd.DataFrame:
    """Create macro features from indicators such as CPI, PMI, and rates."""

    cfg = config or MacroFeatureConfig()
    if macro_data.empty:
        raise ValueError("macro_data cannot be empty")
    if not isinstance(macro_data.index, pd.DatetimeIndex):
        raise TypeError("macro_data must use a DatetimeIndex")

    data = macro_data.sort_index().astype(float)
    features = pd.DataFrame(index=data.index)
    for column in data.columns:
        rolling_mean = data[column].rolling(
            cfg.zscore_window,
            min_periods=max(3, cfg.zscore_window // 3),
        ).mean()
        rolling_std = data[column].rolling(
            cfg.zscore_window,
            min_periods=max(3, cfg.zscore_window // 3),
        ).std(ddof=0)
        features[f"{column}_level"] = data[column]
        features[f"{column}_change"] = data[column].diff(cfg.change_periods)
        features[f"{column}_zscore"] = (
            (data[column] - rolling_mean) / rolling_std.replace(0.0, pd.NA)
        )

    return features.replace([float("inf"), float("-inf")], pd.NA).fillna(0.0)
