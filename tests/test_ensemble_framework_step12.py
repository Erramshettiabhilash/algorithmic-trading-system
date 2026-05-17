"""Tests for Step 12 ensemble framework."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from evaluation import (
    compare_ensemble_to_base_models,
    ensemble_information_coefficient,
    prediction_correlation_matrix,
)
from models import (
    EnsembleConfig,
    ModelPrediction,
    StackingEnsembleModel,
    VotingEnsembleModel,
    WeightedAverageEnsembleModel,
    align_prediction_frame,
    voting_predictions,
    weighted_average_predictions,
)


@dataclass
class _ColumnModel:
    column: str
    scale: float = 1.0
    target_name: str = "target"

    def fit(self, features: pd.DataFrame, target: pd.Series) -> _ColumnModel:
        self.target_name = target.name or self.target_name
        return self

    def predict(self, features: pd.DataFrame) -> ModelPrediction:
        predictions = (features[self.column] * self.scale).rename(f"{self.column}_prediction")
        return ModelPrediction(
            predictions=predictions,
            target_name=self.target_name,
            model_name=f"column_model_{self.column}",
        )


def _dataset(rows: int = 80) -> tuple[pd.DataFrame, pd.Series]:
    index = pd.date_range("2024-01-01", periods=rows, freq="B", tz="UTC")
    first = np.linspace(-1.0, 1.0, rows)
    second = np.sin(np.linspace(0.0, 8.0, rows))
    third = np.cos(np.linspace(0.0, 5.0, rows))
    target = 0.06 * first + 0.03 * second - 0.01 * third
    features = pd.DataFrame(
        {
            "xgboost_signal": first,
            "lstm_signal": second,
            "regime_signal": third,
        },
        index=index,
    )
    return features, pd.Series(target, index=index, name="forward_return_1")


def test_align_and_weighted_average_predictions() -> None:
    index = pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC")
    predictions = {
        "xgb": pd.Series([0.1, 0.2, 0.3, 0.4, 0.5], index=index),
        "lstm": pd.Series([0.0, 0.1, 0.2, 0.3], index=index[1:]),
    }

    frame = align_prediction_frame(predictions)
    combined = weighted_average_predictions(frame, {"xgb": 0.75, "lstm": 0.25})

    assert list(frame.columns) == ["xgb", "lstm"]
    assert frame.index.min() == index[1]
    assert combined.iloc[0] == 0.75 * 0.2 + 0.25 * 0.0


def test_voting_predictions_return_majority_probability() -> None:
    frame = pd.DataFrame(
        {
            "xgb": [0.1, -0.2, 0.3],
            "lstm": [0.2, -0.1, -0.4],
            "regime": [-0.1, -0.3, 0.5],
        }
    )

    votes = voting_predictions(frame)

    assert votes.tolist() == [2 / 3, 0.0, 2 / 3]


def test_weighted_average_ensemble_model_fits_and_predicts() -> None:
    features, target = _dataset()
    ensemble = WeightedAverageEnsembleModel(
        {
            "xgb": lambda: _ColumnModel("xgboost_signal", 0.06),
            "lstm": lambda: _ColumnModel("lstm_signal", 0.03),
        },
        weights={"xgb": 0.7, "lstm": 0.3},
    ).fit(features, target)

    prediction = ensemble.predict(features.tail(10))

    assert prediction.model_name == "weighted_average_ensemble"
    assert prediction.predictions.index.equals(features.tail(10).index)


def test_voting_ensemble_model_outputs_probabilities() -> None:
    features, target = _dataset()
    direction = (target > 0.0).astype(float)
    ensemble = VotingEnsembleModel(
        {
            "xgb": lambda: _ColumnModel("xgboost_signal"),
            "lstm": lambda: _ColumnModel("lstm_signal"),
            "regime": lambda: _ColumnModel("regime_signal", -1.0),
        },
        EnsembleConfig(mode="classification", classification_threshold=0.0),
    ).fit(features, direction)

    prediction = ensemble.predict(features.tail(12))

    assert prediction.predictions.between(0.0, 1.0).all()
    assert prediction.model_name == "voting_ensemble"


def test_stacking_ensemble_uses_meta_model_for_predictions() -> None:
    features, target = _dataset()
    ensemble = StackingEnsembleModel(
        {
            "xgb": lambda: _ColumnModel("xgboost_signal", 0.06),
            "lstm": lambda: _ColumnModel("lstm_signal", 0.03),
            "regime": lambda: _ColumnModel("regime_signal", -0.01),
        },
        EnsembleConfig(mode="regression", meta_test_size=20),
    ).fit(features, target)

    prediction = ensemble.predict(features.tail(15))

    assert prediction.model_name == "stacking_ensemble"
    assert prediction.predictions.index.equals(features.tail(15).index)
    assert np.isfinite(prediction.predictions).all()


def test_ensemble_comparison_and_diversity_metrics() -> None:
    features, target = _dataset()
    base_predictions = {
        "xgb": (features["xgboost_signal"] * 0.06).rename("xgb"),
        "lstm": (features["lstm_signal"] * 0.03).rename("lstm"),
        "regime": (features["regime_signal"] * -0.01).rename("regime"),
    }
    ensemble_prediction = weighted_average_predictions(
        base_predictions,
        {"xgb": 0.5, "lstm": 0.3, "regime": 0.2},
    )

    comparison = compare_ensemble_to_base_models(
        base_predictions,
        ensemble_prediction,
        target,
        ensemble_name="weighted",
    )
    correlations = prediction_correlation_matrix(base_predictions)
    ic = ensemble_information_coefficient(ensemble_prediction, target)

    assert "weighted" in comparison.evaluations.index
    assert comparison.best_base_model in base_predictions
    assert correlations.shape == (3, 3)
    assert -1.0 <= ic <= 1.0
