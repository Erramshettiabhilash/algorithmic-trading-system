# Step 3 - Feature Engineering Engine

## Goal

Convert clean OHLCV data into predictive factors that a machine learning model can consume.
Feature engineering is where raw market data becomes a research hypothesis.

## Why Features Matter

Prices themselves are usually non-stationary. A stock at 100 and a crypto asset at 50,000 cannot be
compared directly, but their returns, volatility, momentum, and volume surprises can. Good features
compress market behavior into signals that are more stable across assets and time.

## Implemented Feature Families

### Return Features

- `log_return_1`
- `rolling_return_N`
- `return_autocorr_N`

Why they matter: return features measure recent performance and short-term persistence or mean
reversion. Autocorrelation helps answer whether recent moves tend to continue or reverse.

### Momentum Features

- RSI
- EMA fast/slow crossover
- MACD line, signal, and histogram
- Momentum factor

Why they matter: momentum is one of the most persistent empirical effects in markets. Trend
followers, CTAs, and many cross-sectional equity strategies all rely on the idea that winners can
keep winning over some horizons.

### Volatility Features

- Rolling annualized volatility
- Average True Range
- ATR ratio
- Volatility regime classification

Why they matter: volatility controls position sizing, stop distance, model confidence, and regime
behavior. A signal that works in low-volatility trends may fail badly in high-volatility chop.

### Volume Features

- On-Balance Volume
- Volume ratio
- Volume z-score

Why they matter: volume helps separate weak moves from institutionally supported moves. Price moves
with abnormal volume often carry more information than price moves on quiet trading.

### Market-Structure Features

- Confirmed fractal highs/lows
- Liquidity sweep highs/lows
- Trend position
- Higher-high/lower-low structure

Why they matter: market-structure features describe how price interacts with recent highs and lows.
They are useful for detecting breakout attempts, stop runs, trend continuation, and failed moves.

## Lookahead-Safe Fractals

Classic fractals require future bars to confirm a local high or low. The implementation shifts the
fractal signal forward to the confirmation timestamp. That means the model only sees a fractal once
it would have been knowable in real time.

## Minimal Example

```python
from pathlib import Path

from data import MarketDataConfig, load_market_data
from features import FeatureConfig, QuantFeatureEngine

config = MarketDataConfig(
    symbol="SAMPLE",
    asset_class="stock",
    source="csv",
    raw_path=Path("data/sample_ohlcv.csv"),
)

prices = load_market_data(config)
demo_config = FeatureConfig(
    return_windows=(1, 3),
    volatility_windows=(3,),
    momentum_windows=(3,),
    rsi_window=3,
    ema_fast=3,
    ema_slow=5,
    macd_signal=2,
    atr_window=3,
    volume_window=3,
    structure_window=3,
)
features = QuantFeatureEngine(demo_config).transform(prices)

print(features.tail())
```

## Interview-Ready Explanation

I group features into return, momentum, volatility, volume, and market-structure families. This
mirrors professional factor research: each group represents a hypothesis about why markets move.
Momentum captures persistence, volatility captures risk regime, volume captures participation, and
market structure captures interactions with recent highs and lows. I also make sure feature
timestamps reflect what would have been known at that time, which is essential for avoiding
lookahead bias.
