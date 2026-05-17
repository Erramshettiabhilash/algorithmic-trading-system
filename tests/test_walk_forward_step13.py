"""Tests for Step 13 walk-forward research pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from evaluation import (
    WalkForwardConfig,
    analyze_alpha_decay,
    analyze_model_stability,
    analyze_prediction_drift,
    run_walk_forward_research,
)
from models import ModelPrediction


@dataclass
class _LinearSignalModel:
    coefficient: float = 0.0
    target_name: str = "target"

    def fit(self, features: pd.DataFrame, target: pd.Series) -> _LinearSignalModel:
        self.target_name = target.name or self.target_name
        signal = features["signal"]
        denominator = float((signal * signal).sum())
        if denominator == 0.0:
            self.coefficient = 0.0
        else:
            self.coefficient = float((signal * target).sum() / denominator)
        return self

    def predict(self, features: pd.DataFrame) -> ModelPrediction:
        predictions = (features["signal"] * self.coefficient).rename("linear_signal_prediction")
        return ModelPrediction(
            predictions=predictions,
            target_name=self.target_name,
            model_name="linear_signal",
        )


def _dataset(rows: int = 90) -> tuple[pd.DataFrame, pd.Series]:
    index = pd.date_range("2024-01-01", periods=rows, freq="B", tz="UTC")
    signal = np.sin(np.linspace(0.0, 10.0, rows))
    drift = np.linspace(-0.2, 0.2, rows)
    target = 0.04 * signal + 0.01 * drift
    features = pd.DataFrame({"signal": signal, "drift": drift}, index=index)
    return features, pd.Series(target, index=index, name="forward_return_1")


def test_expanding_walk_forward_collects_out_of_sample_predictions() -> None:
    features, target = _dataset()
    result = run_walk_forward_research(
        features,
        target,
        model_factory=_LinearSignalModel,
        config=WalkForwardConfig(
            mode="expanding",
            initial_train_size=30,
            test_size=10,
            step_size=10,
        ),
    )

    assert len(result.fold_results) == 6
    assert len(result.predictions) == 60
    assert result.predictions.index.equals(result.realized.index)
    assert result.fold_summary["observations"].sum() == 60
    assert result.evaluation.observations == 60


def test_rolling_walk_forward_uses_fixed_training_window() -> None:
    features, target = _dataset()
    result = run_walk_forward_research(
        features,
        target,
        model_factory=_LinearSignalModel,
        config=WalkForwardConfig(
            mode="rolling",
            train_size=30,
            test_size=15,
            step_size=15,
        ),
    )

    first = result.fold_results[0]
    second = result.fold_results[1]

    assert len(result.fold_results) == 4
    assert first.train_start < second.train_start
    assert first.train_end < second.train_end


def test_alpha_decay_measures_ic_by_horizon() -> None:
    features, target = _dataset()
    predictions = features["signal"].rename("prediction")
    horizons = {
        1: target,
        5: target.shift(-4),
        10: -target,
    }

    decay = analyze_alpha_decay(predictions, horizons)

    assert list(decay.index) == [1, 5, 10]
    assert "alpha_decay" in decay.columns
    assert decay.loc[1, "abs_ic"] >= 0.0


def test_prediction_drift_returns_rolling_diagnostics() -> None:
    index = pd.date_range("2024-01-01", periods=20, freq="B", tz="UTC")
    predictions = pd.Series(np.linspace(-0.1, 0.2, 20), index=index)

    drift = analyze_prediction_drift(predictions, window=5)

    assert {"prediction_mean", "prediction_volatility", "prediction_drift_zscore"}.issubset(
        drift.columns
    )
    assert drift.index.equals(index)
    assert drift.isna().sum().sum() == 0


def test_model_stability_summarizes_fold_metrics() -> None:
    features, target = _dataset()
    result = run_walk_forward_research(
        features,
        target,
        model_factory=_LinearSignalModel,
        config=WalkForwardConfig(initial_train_size=30, test_size=10, step_size=10),
    )

    stability = analyze_model_stability(result.fold_summary)

    assert "mean_ic" in stability.index
    assert "positive_ic_rate" in stability.index
    assert 0.0 <= stability["positive_ic_rate"] <= 1.0
