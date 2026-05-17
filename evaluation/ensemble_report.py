"""Evaluation helpers for ensemble robustness analysis."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import pandas as pd

from evaluation.metrics import information_coefficient
from evaluation.model_report import FactorModelEvaluation, evaluate_factor_predictions


@dataclass(frozen=True)
class EnsembleComparison:
    """Comparison of ensemble metrics against base-model metrics."""

    evaluations: pd.DataFrame
    ensemble_name: str
    best_base_model: str
    sharpe_improvement: float
    ic_improvement: float


def compare_ensemble_to_base_models(
    base_predictions: Mapping[str, pd.Series],
    ensemble_prediction: pd.Series,
    realized_returns: pd.Series,
    *,
    target_type: str = "regression",
    ensemble_name: str = "ensemble",
) -> EnsembleComparison:
    """Evaluate an ensemble against its component prediction streams."""

    rows: dict[str, FactorModelEvaluation] = {}
    for name, predictions in base_predictions.items():
        rows[name] = evaluate_factor_predictions(
            predictions,
            realized_returns,
            target_type=target_type,
        )

    rows[ensemble_name] = evaluate_factor_predictions(
        ensemble_prediction,
        realized_returns,
        target_type=target_type,
    )
    evaluations = pd.DataFrame({name: evaluation.__dict__ for name, evaluation in rows.items()}).T
    base_frame = evaluations.drop(index=ensemble_name)
    best_base_name = str(base_frame["sharpe"].idxmax())
    sharpe_improvement = float(
        evaluations.loc[ensemble_name, "sharpe"] - base_frame["sharpe"].max()
    )
    ic_improvement = float(
        evaluations.loc[ensemble_name, "information_coefficient"]
        - base_frame["information_coefficient"].max()
    )

    return EnsembleComparison(
        evaluations=evaluations,
        ensemble_name=ensemble_name,
        best_base_model=best_base_name,
        sharpe_improvement=sharpe_improvement,
        ic_improvement=ic_improvement,
    )


def prediction_correlation_matrix(predictions: Mapping[str, pd.Series]) -> pd.DataFrame:
    """Measure prediction diversity across base models."""

    if not predictions:
        raise ValueError("at least one prediction series is required")

    frame = pd.concat([series.rename(name) for name, series in predictions.items()], axis=1)
    return frame.dropna().corr()


def ensemble_information_coefficient(
    ensemble_prediction: pd.Series,
    realized_returns: pd.Series,
) -> float:
    """Convenience wrapper for ensemble IC reporting."""

    return information_coefficient(ensemble_prediction, realized_returns)
