# Step 2 - Data Collection and Preprocessing

## Goal

Build a financial data pipeline that can ingest OHLCV bars from CSV files or yfinance and convert
them into clean, aligned, feature-ready datasets.

## Why Data Engineering Matters In Finance

Financial ML is unusually sensitive to small data mistakes. A single timestamp mismatch can turn a
bad model into a fake alpha model. Before feature engineering or XGBoost, we need trustworthy bars.

## Assets Covered

- Stocks: examples include `AAPL`, `MSFT`, `SPY`.
- Forex: examples include `EURUSD=X`, `GBPUSD=X`.
- Gold: examples include `GC=F` or gold ETFs such as `GLD`.
- Crypto: examples include `BTC-USD`, `ETH-USD`.

## Implemented Pipeline Pieces

- `data.sources.load_csv_ohlcv`: reads local CSV datasets.
- `data.sources.fetch_yfinance_ohlcv`: fetches yfinance bars with lazy import.
- `data.sources.load_market_data`: dispatches based on `MarketDataConfig`.
- `data.preprocessing.clean_ohlcv`: standardizes schema, aligns timestamps, handles missing
  values, validates OHLC consistency.
- `data.preprocessing.align_multi_asset_ohlcv`: aligns multiple assets on shared timestamps.
- `data.preprocessing.create_normalized_ohlcv_features`: creates normalized, feature-ready
  return/range/volume columns.
- `data.preprocessing.build_feature_ready_dataset`: produces multi-asset model input with
  explicit asset and feature column levels.

## Important Biases

Lookahead bias: using information that was not available at the decision timestamp. In the Step 2
pipeline, missing prices are forward-filled only. Backward filling is avoided because it injects
future data into the past.

Survivorship bias: testing only on assets that still exist today. For institutional research, the
asset universe must include delisted and failed names where relevant.

Data leakage: letting target-period information enter training features. A classic example is
normalizing the full dataset before a temporal split. Later steps will fit scalers only on training
windows.

Temporal consistency: all joins, fills, features, and targets must respect chronological order. In
finance, random row shuffling is usually wrong because yesterday and tomorrow are not independent.

## Minimal Example

```python
from pathlib import Path

from data import (
    MarketDataConfig,
    build_feature_ready_dataset,
    load_market_data,
)

config = MarketDataConfig(
    symbol="SAMPLE",
    asset_class="stock",
    source="csv",
    raw_path=Path("data/sample_ohlcv.csv"),
)

prices = load_market_data(config)
dataset = build_feature_ready_dataset({"SAMPLE": prices})

print(dataset.tail())
```

## Interview-Ready Explanation

Before modeling, I standardize all market data into a common OHLCV schema with timezone-aware
timestamps. I clean missing values using time-safe rules, align assets on common timestamps, and
convert raw prices into normalized return-style columns. This reduces leakage risk and gives every
downstream model the same reproducible input contract.
