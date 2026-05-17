"""Starter data pipeline primitives for OHLCV market datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

REQUIRED_OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")


@dataclass(frozen=True)
class MarketDataConfig:
    """Configuration for loading a single asset time series."""

    symbol: str
    asset_class: str
    source: str = "csv"
    timezone: str = "UTC"
    raw_path: Path | None = None
    timestamp_column: str = "date"
    interval: str = "1d"


def normalize_ohlcv_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Return an OHLCV frame with normalized lowercase column names.

    Professional research code standardizes column names early so every downstream
    feature, model, and backtest consumes the same schema.
    """

    normalized = frame.rename(columns={column: column.strip().lower() for column in frame.columns})
    missing = [column for column in REQUIRED_OHLCV_COLUMNS if column not in normalized.columns]
    if missing:
        raise ValueError(f"missing required OHLCV columns: {missing}")

    return normalized.loc[:, list(REQUIRED_OHLCV_COLUMNS)].copy()


def align_time_index(frame: pd.DataFrame, timezone: str = "UTC") -> pd.DataFrame:
    """Return a chronologically sorted, timezone-aware OHLCV frame."""

    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError("market data index must be a pandas DatetimeIndex")

    aligned = frame.sort_index()
    if aligned.index.tz is None:
        aligned.index = aligned.index.tz_localize(timezone)
    else:
        aligned.index = aligned.index.tz_convert(timezone)

    return aligned[~aligned.index.duplicated(keep="last")]
