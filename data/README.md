# Data

This folder stores example market datasets and data-pipeline code for the quant research
platform.

## Data Layout

Recommended layout for real projects:

```text
data/
|-- raw/              # original vendor files, usually not committed
|-- processed/        # cleaned and aligned research datasets, usually not committed
|-- sample_ohlcv.csv  # small committed example dataset
`-- sample_vol_surface.csv
```

Large raw market data files should stay out of Git. Commit only small samples that make tests,
examples, and documentation reproducible.

## Standard OHLCV Schema

All source adapters should return:

```text
open, high, low, close, volume
```

with a timezone-aware `DatetimeIndex`.

Why this matters: every feature, model, and backtest should consume the same schema. If one module
expects `Close` and another expects `close`, research code becomes fragile and silent mistakes
become likely.

## Supported Step 2 Sources

- CSV files via `load_csv_ohlcv`
- yfinance via `fetch_yfinance_ohlcv`
- source-dispatched loading via `load_market_data`

Alpha Vantage support is intentionally left as a later adapter because it needs an API key and
rate-limit handling.

## Finance Data Pitfalls

Lookahead bias happens when future information enters a historical decision. Example: backward
filling a missing price or using tomorrow's close to build today's signal.

Survivorship bias happens when the dataset includes only assets that survived until today. A stock
strategy tested only on current index members usually overstates performance because bankrupt or
delisted names disappeared from the universe.

Data leakage happens when model inputs contain information from the target period. Example: scaling
the entire dataset before a train/test split, or computing features using future rows.

Temporal consistency means every observation must represent what was knowable at that timestamp.
In quant research, time order is part of the data contract.
