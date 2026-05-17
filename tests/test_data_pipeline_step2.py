"""Tests for Step 2 market data ingestion and preprocessing."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from data import (
    MarketDataConfig,
    align_multi_asset_ohlcv,
    build_feature_ready_dataset,
    clean_ohlcv,
    create_normalized_ohlcv_features,
    load_csv_ohlcv,
    load_market_data,
)


def _raw_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Open": [100.0, None, 101.0, 102.0, 101.0, 103.0],
            "High": [101.0, 101.5, 102.0, 103.0, 102.5, 104.0],
            "Low": [99.5, 100.0, 100.5, 101.0, 100.5, 102.0],
            "Close": [100.5, 101.0, 101.5, 102.5, 101.2, 103.5],
            "Volume": [1_000, None, 1_200, 1_400, 1_300, 1_500],
        },
        index=pd.to_datetime(
            [
                "2024-01-02",
                "2024-01-03",
                "2024-01-04",
                "2024-01-05",
                "2024-01-08",
                "2024-01-09",
            ]
        ),
    )


def test_csv_loader_returns_standard_schema() -> None:
    frame = load_csv_ohlcv(Path("data/sample_ohlcv.csv"))

    assert list(frame.columns) == ["open", "high", "low", "close", "volume"]
    assert frame.index.is_monotonic_increasing
    assert frame.index.tz is not None


def test_market_data_config_loads_csv_source() -> None:
    config = MarketDataConfig(
        symbol="SAMPLE",
        asset_class="stock",
        source="csv",
        raw_path=Path("data/sample_ohlcv.csv"),
    )

    frame = load_market_data(config)

    assert frame["close"].iloc[-1] == 104.8


def test_clean_ohlcv_forward_fills_without_backward_leakage() -> None:
    clean = clean_ohlcv(_raw_frame())

    assert clean["open"].iloc[1] == 100.0
    assert clean["volume"].iloc[1] == 0.0
    assert clean.index.tz is not None


def test_clean_ohlcv_rejects_inconsistent_bars() -> None:
    bad = _raw_frame()
    bad.loc[bad.index[0], "High"] = 99.0

    with pytest.raises(ValueError, match="OHLC bounds"):
        clean_ohlcv(bad)


def test_normalized_features_are_feature_ready() -> None:
    clean = clean_ohlcv(_raw_frame())
    features = create_normalized_ohlcv_features(clean)

    assert "log_return" in features.columns
    assert "volume_zscore" in features.columns
    assert features.isna().sum().sum() == 0


def test_multi_asset_alignment_uses_shared_timestamps() -> None:
    first = _raw_frame()
    second = _raw_frame().iloc[1:].copy()

    aligned = align_multi_asset_ohlcv({"STOCK": first, "CRYPTO": second})

    assert aligned.columns.nlevels == 2
    assert aligned.index.min() == clean_ohlcv(second).index.min()


def test_feature_ready_dataset_keeps_asset_level() -> None:
    first = _raw_frame()
    second = _raw_frame()
    for column in ["Open", "High", "Low", "Close"]:
        second[column] = second[column] * 1.01

    dataset = build_feature_ready_dataset({"GOLD": first, "FX": second})

    assert dataset.columns.nlevels == 2
    assert ("GOLD", "log_return") in dataset.columns
