"""Preprocessing utilities for leakage-safe financial time-series research."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from data.pipeline import REQUIRED_OHLCV_COLUMNS, align_time_index, normalize_ohlcv_columns

PRICE_COLUMNS = ("open", "high", "low", "close")


def clean_ohlcv(
    frame: pd.DataFrame,
    *,
    timezone: str = "UTC",
    fill_method: str = "ffill",
    drop_remaining_na: bool = True,
) -> pd.DataFrame:
    """Clean one OHLCV frame while preserving temporal order.

    The function only uses past observations for forward filling. That matters because
    backward filling would let future prices leak into earlier timestamps.
    """

    clean = normalize_ohlcv_columns(frame)
    clean = align_time_index(clean, timezone=timezone)
    clean = clean.apply(pd.to_numeric, errors="coerce")
    clean = clean.replace([np.inf, -np.inf], np.nan)

    if fill_method == "ffill":
        clean.loc[:, PRICE_COLUMNS] = clean.loc[:, PRICE_COLUMNS].ffill()
        clean.loc[:, "volume"] = clean.loc[:, "volume"].fillna(0.0)
    elif fill_method == "none":
        pass
    else:
        raise ValueError("fill_method must be either 'ffill' or 'none'")

    if drop_remaining_na:
        clean = clean.dropna()

    clean = clean[(clean.loc[:, PRICE_COLUMNS] > 0).all(axis=1)]
    clean = clean[clean["volume"] >= 0]
    _validate_ohlc_bounds(clean)
    return clean


def create_normalized_ohlcv_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create stationary, model-ready normalized columns from clean OHLCV data.

    Raw prices are non-stationary; returns and relative ranges are usually more useful
    model inputs because they describe changes rather than absolute price levels.
    """

    clean = clean_ohlcv(frame)
    output = clean.copy()
    output["close_return"] = output["close"].pct_change()
    output["log_return"] = np.log(output["close"]).diff()
    output["high_low_range"] = output["high"] / output["low"] - 1.0
    output["close_open_return"] = output["close"] / output["open"] - 1.0

    volume_mean = output["volume"].rolling(20, min_periods=5).mean()
    volume_std = output["volume"].rolling(20, min_periods=5).std(ddof=0)
    output["volume_zscore"] = (output["volume"] - volume_mean) / volume_std.replace(0.0, np.nan)

    return output.dropna()


def align_multi_asset_ohlcv(
    assets: Mapping[str, pd.DataFrame],
    *,
    timezone: str = "UTC",
    join: str = "inner",
) -> pd.DataFrame:
    """Align multiple assets onto one timestamp index with MultiIndex columns."""

    if not assets:
        raise ValueError("at least one asset frame is required")

    aligned_assets: dict[str, pd.DataFrame] = {}
    for symbol, frame in assets.items():
        aligned_assets[symbol] = clean_ohlcv(frame, timezone=timezone)

    aligned = pd.concat(aligned_assets, axis=1, join=join).sort_index()
    if join == "inner":
        aligned = aligned.dropna()
    elif join != "outer":
        raise ValueError("join must be either 'inner' or 'outer'")

    return aligned


def build_feature_ready_dataset(
    assets: Mapping[str, pd.DataFrame],
    *,
    timezone: str = "UTC",
    join: str = "inner",
) -> pd.DataFrame:
    """Return aligned normalized datasets for multiple assets.

    The output keeps the first column level as the asset symbol and the second level as
    the normalized feature name, which makes cross-asset modeling and slicing explicit.
    """

    if not assets:
        raise ValueError("at least one asset frame is required")

    feature_frames: dict[str, pd.DataFrame] = {}
    for symbol, frame in assets.items():
        clean = clean_ohlcv(frame, timezone=timezone)
        feature_frames[symbol] = create_normalized_ohlcv_features(clean)

    dataset = pd.concat(feature_frames, axis=1, join=join).sort_index()
    if join == "inner":
        dataset = dataset.dropna()
    elif join != "outer":
        raise ValueError("join must be either 'inner' or 'outer'")

    return dataset


def _validate_ohlc_bounds(frame: pd.DataFrame) -> None:
    """Raise when OHLC bars violate basic market-data consistency."""

    if frame.empty:
        raise ValueError("cleaned OHLCV frame is empty")

    high_is_valid = frame["high"] >= frame.loc[:, REQUIRED_OHLCV_COLUMNS[:4]].max(axis=1)
    low_is_valid = frame["low"] <= frame.loc[:, REQUIRED_OHLCV_COLUMNS[:4]].min(axis=1)
    if not bool(high_is_valid.all() and low_is_valid.all()):
        raise ValueError("OHLC bounds are inconsistent: high/low do not contain open/close")
