# Step 7 - SHAP Explainability Engine

## Goal

Explain why a model made a prediction. In finance, this is not cosmetic. A trading model affects
risk, capital allocation, and portfolio decisions, so its behavior must be inspectable.

## Why Black-Box Models Are Dangerous In Finance

A model can look strong in backtests while relying on unstable or accidental relationships. For
example:

- It may overreact to volatility because the training period had one crisis.
- It may treat volume spikes as bullish in one regime and bearish in another.
- It may learn a leakage feature that looks predictive only because of data alignment mistakes.
- It may perform well statistically but be impossible to defend to a risk committee.

Explainability helps identify whether the model is using sensible signals or fragile artifacts.

## Implemented Components

- `ShapExplainabilityEngine.explain`: computes SHAP values.
- `feature_importance`: ranks features by mean absolute SHAP value.
- `dependence_data`: returns feature value vs SHAP contribution data.
- `local_explanation`: explains one timestamp/prediction.
- `interaction_summary`: summarizes approximate interaction strength.
- `save_summary_plot`: saves a SHAP summary plot.
- `interpret_feature_effect`: converts one contribution into finance-readable language.

## Feature Importance

Mean absolute SHAP value answers:

```text
Which features moved predictions the most on average?
```

This is stronger than basic tree importance because SHAP shows the contribution to actual model
outputs.

## Local Explanation

A local explanation answers:

```text
For this timestamp, which factors pushed the forecast up or down?
```

Example:

```python
from explainability import ShapExplainabilityEngine

engine = ShapExplainabilityEngine(model)
explanation = engine.explain(x_test)
local = engine.local_explanation(explanation, row=-1)

print(local.contributions)
```

## Dependence Plots

Dependence data shows how a feature's value relates to its SHAP contribution. This helps answer:

- Does higher momentum consistently increase predicted returns?
- Does high volatility reduce model confidence?
- Is the relationship nonlinear?

## Feature Interaction Analysis

The current implementation includes an approximate interaction summary based on SHAP contribution
correlations. Later, for tree models, this can be expanded to native SHAP interaction values.

Useful finance questions:

- Does momentum matter only when volatility is low?
- Does volume confirmation strengthen breakout signals?
- Do market-structure signals behave differently in high-risk regimes?

## Interview-Ready Interpretation Example

If `momentum_factor_10` has a positive SHAP value, you can say:

```text
The model increased its return forecast because recent momentum was supportive. This suggests the
tree ensemble learned a trend-following relationship for this observation.
```

If `rolling_volatility_20` has a negative SHAP value:

```text
The model reduced its forecast because volatility was elevated. In trading terms, the model is
penalizing riskier regimes where signals may be less reliable.
```

Important: SHAP explains model behavior, not true causality. It tells us what moved the model's
prediction, not what caused the market return.

## Minimal Example

```python
from explainability import ShapExplainabilityEngine, interpret_feature_effect

engine = ShapExplainabilityEngine(model)
explanation = engine.explain(x_test)

importance = engine.feature_importance(explanation)
dependence = engine.dependence_data(explanation, "momentum_factor_10")
local = engine.local_explanation(explanation, row=-1)

for feature, value in local.contributions.items():
    print(interpret_feature_effect(feature, value))
```

## Governance Takeaway

Explainability is part of model risk management. It helps researchers detect leakage, confirm
economic intuition, document model behavior, and communicate risks before a model is connected to a
trading system.
