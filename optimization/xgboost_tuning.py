"""Small temporal hyperparameter tuning helpers for XGBoost factor models."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import pandas as pd

from evaluation.model_report import FactorModelEvaluation, evaluate_factor_predictions
from evaluation.validation import temporal_train_test_split
from models.xgboost_factor import XGBoostFactorModel, XGBoostModelConfig


@dataclass(frozen=True)
class XGBoostTuningResult:
    """Best temporal-validation result from a parameter search."""

    config: XGBoostModelConfig
    evaluation: FactorModelEvaluation


def tune_xgboost_factor_model(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    candidate_configs: Iterable[XGBoostModelConfig],
    test_size: float | int = 0.2,
) -> XGBoostTuningResult:
    """Select the best XGBoost configuration using a chronological validation split."""

    split = temporal_train_test_split(features, target, test_size=test_size)
    best: XGBoostTuningResult | None = None

    for config in candidate_configs:
        model = XGBoostFactorModel(config).fit(split.x_train, split.y_train)
        predictions = model.predict(split.x_test).predictions
        evaluation = evaluate_factor_predictions(
            predictions,
            split.y_test,
            target_type=config.objective_type,
        )
        result = XGBoostTuningResult(config=config, evaluation=evaluation)

        if best is None or _score(result) > _score(best):
            best = result

    if best is None:
        raise ValueError("candidate_configs must contain at least one configuration")

    return best


def _score(result: XGBoostTuningResult) -> float:
    """Rank candidates by IC first, then lower RMSE."""

    return result.evaluation.information_coefficient - result.evaluation.rmse
