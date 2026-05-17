# Step 5 - XGBoost Factor Model

## Goal

Train the first machine learning model in the platform: an XGBoost factor model that predicts
future returns or market direction from engineered features.

## Why XGBoost Is Popular In Quant Finance

XGBoost works well on tabular factor data. Most quant research datasets are not images or raw text;
they are rows of engineered features such as momentum, volatility, volume, and valuation factors.
Tree boosting is strong in this setting because it captures nonlinear relationships and feature
interactions without requiring heavy feature scaling.

Examples:

- Momentum may work only when volatility is low.
- Volume spikes may matter only near recent highs or lows.
- Mean reversion may appear after unusually large short-term returns.

Linear models struggle with these conditional effects unless we manually create interaction terms.
XGBoost can discover many of them directly.

## Implemented Components

- `models.xgboost_factor.XGBoostModelConfig`: model configuration.
- `models.xgboost_factor.XGBoostFactorModel`: regression/classification wrapper.
- `optimization.xgboost_tuning.tune_xgboost_factor_model`: simple chronological validation search.
- `evaluation.model_report.evaluate_factor_predictions`: combined ML/trading report.

## Regression Workflow

Regression predicts future return magnitude.

```python
from evaluation import create_forward_return_target, build_supervised_dataset
from models import XGBoostFactorModel, XGBoostModelConfig

target = create_forward_return_target(prices, horizon=1)
x, y = build_supervised_dataset(features, target)

model = XGBoostFactorModel(
    XGBoostModelConfig(objective_type="regression")
).fit(x, y)

predictions = model.predict(x).predictions
```

Regression is useful when predictions will be ranked or converted into position sizes.

## Classification Workflow

Classification predicts whether the future return is positive.

```python
from evaluation import create_direction_target, build_supervised_dataset
from models import XGBoostFactorModel, XGBoostModelConfig

target = create_direction_target(prices, horizon=1)
x, y = build_supervised_dataset(features, target)

model = XGBoostFactorModel(
    XGBoostModelConfig(objective_type="classification")
).fit(x, y)

probability_up = model.predict(x).predictions
```

Classification is useful when the strategy only needs a long/short/flat decision.

## Evaluation Metrics

RMSE measures prediction error for return magnitude.

Accuracy measures directional correctness. It is intuitive, but dangerous if used alone.

Sharpe Ratio measures risk-adjusted strategy returns.

Max Drawdown measures worst peak-to-trough loss.

Information Coefficient measures whether predictions rank future returns correctly.

## Why Accuracy Alone Is Not Enough

A model can be 55% accurate and still lose money if the 45% wrong trades are much larger than the
correct trades. In trading, we care about payoff distribution, not just hit rate. That is why the
Step 5 report combines ML metrics with trading metrics.

## Minimal Temporal Tuning Example

```python
from models import XGBoostModelConfig
from optimization import tune_xgboost_factor_model

candidates = [
    XGBoostModelConfig(n_estimators=50, max_depth=2, learning_rate=0.05),
    XGBoostModelConfig(n_estimators=100, max_depth=3, learning_rate=0.03),
]

result = tune_xgboost_factor_model(
    x,
    y,
    candidate_configs=candidates,
    test_size=0.2,
)

print(result.config)
print(result.evaluation)
```

## Interview-Ready Explanation

I use XGBoost as the first factor model because it performs well on structured tabular financial
features and can capture nonlinear feature interactions. I train it using chronological validation,
not random shuffling, then evaluate both prediction quality and trading quality. That distinction
is important because a model with decent RMSE or accuracy may still have poor Sharpe, large
drawdowns, or weak IC.
