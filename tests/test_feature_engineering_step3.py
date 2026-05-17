"""Tests for Step 3 quantitative feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd

from features import (
    FeatureConfig,
    QuantFeatureEngine,
    build_multi_asset_feature_panel,
    relative_strength_index,
)
from features.market_structure import confirmed_fractal_high, confirmed_fractal_low


def _price_frame(rows: int = 90) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=rows, freq="B", tz="UTC")
    trend = np.linspace(100.0, 125.0, rows)
    cycle = 2.0 * np.sin(np.linspace(0.0, 8.0, rows))
    close = trend + cycle
    open_ = close * (1.0 + 0.001 * np.cos(np.linspace(0.0, 6.0, rows)))
    high = np.maximum(open_, close) + 0.75
    low = np.minimum(open_, close) - 0.75
    volume = 1_000_000 + 50_000 * np.sin(np.linspace(0.0, 10.0, rows))

    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=dates,
    )


def test_quant_feature_engine_creates_expected_factor_families() -> None:
    engine = QuantFeatureEngine()
    features = engine.transform(_price_frame())

    expected_columns = {
        "log_return_1",
        "return_autocorr_10",
        "rsi_14",
        "ema_crossover_12_26",
        "macd_histogram",
        "rolling_volatility_20",
        "atr_ratio_14",
        "obv",
        "volume_zscore_20",
        "confirmed_fractal_high_2",
        "liquidity_sweep_high_20",
        "trend_structure_20",
    }

    assert expected_columns.issubset(features.columns)
    assert features.isna().sum().sum() == 0
    assert features.index.is_monotonic_increasing


def test_feature_engine_can_keep_warmup_rows_when_requested() -> None:
    config = FeatureConfig(drop_na=False)
    features = QuantFeatureEngine(config).transform(_price_frame())

    assert len(features) == 90


def test_rsi_stays_in_valid_range_after_warmup() -> None:
    close = _price_frame()["close"]
    rsi = relative_strength_index(close).dropna()

    assert ((rsi >= 0.0) & (rsi <= 100.0)).all()


def test_confirmed_fractals_are_shifted_to_confirmation_time() -> None:
    high = pd.Series([1.0, 2.0, 5.0, 2.0, 1.0, 3.0])
    low = pd.Series([5.0, 4.0, 1.0, 4.0, 5.0, 3.0])

    high_fractal = confirmed_fractal_high(high, order=2)
    low_fractal = confirmed_fractal_low(low, order=2)

    assert high_fractal.iloc[4] == 1.0
    assert low_fractal.iloc[4] == 1.0


def test_multi_asset_feature_panel_uses_asset_column_level() -> None:
    first = _price_frame()
    second = _price_frame().assign(
        open=lambda frame: frame["open"] * 1.02,
        high=lambda frame: frame["high"] * 1.02,
        low=lambda frame: frame["low"] * 1.02,
        close=lambda frame: frame["close"] * 1.02,
    )

    panel = build_multi_asset_feature_panel({"STOCK": first, "CRYPTO": second})

    assert panel.columns.nlevels == 2
    assert ("STOCK", "log_return_1") in panel.columns
    assert ("CRYPTO", "macd_histogram") in panel.columns
