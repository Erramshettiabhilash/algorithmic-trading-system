"""Prediction evaluation reports for factor models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from evaluation.metrics import information_coefficient, max_drawdown, sharpe_ratio
from visualization.research_charts import prepare_equity_curve


@dataclass(frozen=True)
class FactorModelEvaluation:
    """Combined ML and trading evaluation for one prediction series."""

    rmse: float
    accuracy: float
    sharpe: float
    max_drawdown: float
    information_coefficient: float
    observations: int


def evaluate_factor_predictions(
    predictions: pd.Series,
    realized_returns: pd.Series,
    *,
    target_type: str = "regression",
    classification_threshold: float = 0.5,
    periods_per_year: int = 252,
) -> FactorModelEvaluation:
    """Evaluate predictions using both statistical and trading metrics."""

    aligned = pd.concat(
        [predictions.rename("prediction"), realized_returns.rename("realized_return")],
        axis=1,
    ).dropna()
    if aligned.empty:
        raise ValueError("no aligned prediction/return observations available")

    prediction = aligned["prediction"]
    realized = aligned["realized_return"]

    rmse = float(np.sqrt(np.mean(np.square(prediction - realized))))
    if target_type == "classification":
        predicted_direction = prediction > classification_threshold
    elif target_type == "regression":
        predicted_direction = prediction > 0.0
    else:
        raise ValueError("target_type must be either 'regression' or 'classification'")

    realized_direction = realized > 0.0
    accuracy = float((predicted_direction == realized_direction).mean())

    position = predicted_direction.astype(float).replace({0.0: -1.0})
    strategy_returns = position * realized
    equity_curve = prepare_equity_curve(strategy_returns)

    return FactorModelEvaluation(
        rmse=rmse,
        accuracy=accuracy,
        sharpe=sharpe_ratio(strategy_returns, periods_per_year=periods_per_year),
        max_drawdown=max_drawdown(equity_curve),
        information_coefficient=information_coefficient(prediction, realized),
        observations=len(aligned),
    )
