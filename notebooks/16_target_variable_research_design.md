# Step 4 - Target Variable and Research Design

## Goal

Define exactly what the model is trying to predict and how we evaluate it through time. In finance,
the target and validation design are as important as the algorithm.

## Targets Implemented

### Next-Period Return

`create_forward_return_target` builds a regression target:

```text
return from t to t + horizon
```

The target is stored at timestamp `t`, meaning features known at `t` can be used to predict the
future return without lookahead.

### Market Direction

`create_direction_target` builds a classification target:

```text
1 if future return > threshold, otherwise 0
```

The threshold can represent noise, spread, or transaction costs. For example, predicting a 0.01%
move may be useless if trading costs are 0.05%.

## Regression vs Classification

Regression predicts return magnitude. It is useful for ranking assets, sizing positions, and
building portfolio weights. It is also noisier because exact returns are hard to forecast.

Classification predicts direction. It is often more stable and easier to evaluate with accuracy,
precision, and recall, but it loses magnitude information. A model can be directionally correct and
still lose money if wins are small and losses are large.

## Signal Horizon

The forecast horizon controls what kind of strategy we are researching:

- 1 bar: short-term timing and micro mean reversion.
- 5 to 20 bars: swing signals and medium-term momentum.
- 60+ bars: slower factor or allocation signals.

The horizon must match the holding period, turnover tolerance, and trading costs.

## Alpha Decay

Alpha decay means a signal weakens as time passes. A strong one-day signal may have no value after
ten days. Later walk-forward analysis will measure whether IC falls as the prediction horizon
extends.

## Prediction Stability

Stable signals should retain a similar direction of IC across time windows. A model with one
excellent test period and many weak periods is usually less useful than a model with modest but
consistent IC.

## Temporal Splits

`temporal_train_test_split` trains on the past and tests on the future. It never shuffles rows.

`expanding_window_splits` grows the training set through time. This is useful when old data remains
relevant.

`rolling_window_splits` keeps a fixed training length. This is useful when markets change and old
data becomes stale.

## Why Random Train/Test Split Is Wrong In Finance

Random splits let future market regimes leak into training. If 2024 data appears in training and
2020 data appears in testing, the model has indirectly learned about future distributions. That is
not how trading works. A live strategy always trains on the past and faces the unknown future.

## IC and IR

Information Coefficient measures rank correlation between predictions and realized future returns.
It answers: did the model rank better opportunities above worse opportunities?

Information Ratio measures signal consistency. For an IC series, ICIR is:

```text
mean(IC) / std(IC)
```

A small but stable IC can be more valuable than a large but unstable IC.

## Minimal Example

```python
from evaluation import (
    build_supervised_dataset,
    create_forward_return_target,
    temporal_train_test_split,
)

target = create_forward_return_target(prices, horizon=1)
x, y = build_supervised_dataset(features, target)
split = temporal_train_test_split(x, y, test_size=0.2)

print(split.x_train.shape, split.x_test.shape)
```

## Interview-Ready Explanation

I define targets as forward returns and direction labels aligned to the decision timestamp. Then I
evaluate models with chronological splits, expanding windows, and rolling windows instead of random
train/test splits. This mirrors live trading: train on past data, test on future data, and measure
whether predictions rank future returns using IC and whether that IC is stable using IR.
