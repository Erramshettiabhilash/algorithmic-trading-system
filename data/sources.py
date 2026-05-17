"""Market data source adapters.

The rest of the platform should not care whether data came from a CSV file, yfinance,
Alpha Vantage, or a paid vendor. Source adapters convert vendor output into the same
OHLCV schema before preprocessing begins.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from data.pipeline import MarketDataConfig, align_time_index, normalize_ohlcv_columns


def load_csv_ohlcv(
    path: str | Path,
    *,
    timestamp_column: str = "date",
    timezone: str = "UTC",
) -> pd.DataFrame:
    """Load OHLCV data from a CSV file into the standard market-data schema."""

    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV data file does not exist: {csv_path}")

    raw = pd.read_csv(csv_path)
    if timestamp_column not in raw.columns:
        raise ValueError(f"timestamp column '{timestamp_column}' not found in {csv_path}")

    raw[timestamp_column] = pd.to_datetime(raw[timestamp_column], utc=False)
    raw = raw.set_index(timestamp_column)

    normalized = normalize_ohlcv_columns(raw)
    return align_time_index(normalized, timezone=timezone)


def fetch_yfinance_ohlcv(
    symbol: str,
    *,
    start: str,
    end: str | None = None,
    interval: str = "1d",
    timezone: str = "UTC",
) -> pd.DataFrame:
    """Fetch OHLCV bars from yfinance.

    This function imports yfinance lazily so local tests can run without network access.
    """

    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError("Install yfinance to fetch live market data.") from exc

    raw = yf.download(
        symbol,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if raw.empty:
        raise ValueError(f"yfinance returned no rows for symbol '{symbol}'")

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    normalized = normalize_ohlcv_columns(raw)
    return align_time_index(normalized, timezone=timezone)


def load_market_data(
    config: MarketDataConfig,
    *,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Load one asset using a source selected by ``MarketDataConfig``."""

    source = config.source.lower()
    if source == "csv":
        if config.raw_path is None:
            raise ValueError("raw_path is required when source='csv'")
        return load_csv_ohlcv(
            config.raw_path,
            timestamp_column=config.timestamp_column,
            timezone=config.timezone,
        )
    if source == "yfinance":
        if start is None:
            raise ValueError("start date is required when source='yfinance'")
        return fetch_yfinance_ohlcv(
            config.symbol,
            start=start,
            end=end,
            interval=config.interval,
            timezone=config.timezone,
        )

    raise ValueError(f"unsupported data source: {config.source}")
