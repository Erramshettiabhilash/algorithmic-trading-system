"""Ensemble models and prediction-combination utilities."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from evaluation.validation import temporal_train_test_split
from models.research import ModelPrediction, PredictiveModel

EnsembleMode = Literal["regression", "classification"]
ModelFactory = Callable[[], PredictiveModel]


@dataclass(frozen=True)
class EnsembleConfig:
    """Configuration shared by ensemble models."""

    mode: EnsembleMode = "regression"
    classification_threshold: float = 0.5
    meta_test_size: float | int = 0.25


def align_prediction_frame(predictions: Mapping[str, pd.Series]) -> pd.DataFrame:
    """Align model prediction series onto their shared timestamps."""

    if not predictions:
        raise ValueError("at least one prediction series is required")

    renamed = [series.rename(name) for name, series in predictions.items()]
    frame = pd.concat(renamed, axis=1, join="inner").dropna()
    if frame.empty:
        raise ValueError("prediction series have no shared non-missing timestamps")
    return frame.sort_index()


def weighted_average_predictions(
    predictions: Mapping[str, pd.Series] | pd.DataFrame,
    weights: Mapping[str, float] | None = None,
) -> pd.Series:
    """Combine predictions using normalized weighted averaging."""

    if isinstance(predictions, pd.DataFrame):
        frame = predictions
    else:
        frame = align_prediction_frame(predictions)
    if weights is None:
        weight_vector = pd.Series(1.0, index=frame.columns)
    else:
        missing = set(frame.columns).difference(weights)
        if missing:
            raise ValueError(f"missing weights for models: {sorted(missing)}")
        weight_vector = pd.Series(weights, index=frame.columns, dtype=float)

    total_weight = weight_vector.sum()
    if total_weight == 0.0:
        raise ValueError("sum of ensemble weights cannot be zero")

    normalized = weight_vector / total_weight
    combined = frame.mul(normalized, axis=1).sum(axis=1)
    combined.name = "weighted_ensemble_prediction"
    return combined


def voting_predictions(
    predictions: Mapping[str, pd.Series] | pd.DataFrame,
    *,
    threshold: float = 0.0,
) -> pd.Series:
    """Combine directional model outputs with majority voting."""

    if isinstance(predictions, pd.DataFrame):
        frame = predictions
    else:
        frame = align_prediction_frame(predictions)
    votes = (frame > threshold).astype(float)
    majority_probability = votes.mean(axis=1)
    majority_probability.name = "voting_ensemble_probability"
    return majority_probability


class WeightedAverageEnsembleModel:
    """Fit multiple predictive models and average their predictions."""

    def __init__(
        self,
        model_factories: Mapping[str, ModelFactory],
        weights: Mapping[str, float] | None = None,
        config: EnsembleConfig | None = None,
    ) -> None:
        self.model_factories = dict(model_factories)
        self.weights = dict(weights) if weights is not None else None
        self.config = config or EnsembleConfig()
        self.models: dict[str, PredictiveModel] = {}
        self.target_name = "target"

    def fit(self, features: pd.DataFrame, target: pd.Series) -> WeightedAverageEnsembleModel:
        """Fit every base model on the same chronological training sample."""

        if not self.model_factories:
            raise ValueError("at least one model factory is required")

        self.target_name = target.name or self.target_name
        self.models = {}
        for name, factory in self.model_factories.items():
            self.models[name] = factory().fit(features, target)
        return self

    def predict(self, features: pd.DataFrame) -> ModelPrediction:
        """Generate weighted-average ensemble predictions."""

        if not self.models:
            raise ValueError("ensemble must be fit before predict")

        base_predictions = {
            name: model.predict(features).predictions for name, model in self.models.items()
        }
        combined = weighted_average_predictions(base_predictions, self.weights)
        return ModelPrediction(
            predictions=combined,
            target_name=self.target_name,
            model_name="weighted_average_ensemble",
        )


class VotingEnsembleModel:
    """Fit multiple classifiers or directional regressors and vote on direction."""

    def __init__(
        self,
        model_factories: Mapping[str, ModelFactory],
        config: EnsembleConfig | None = None,
    ) -> None:
        self.model_factories = dict(model_factories)
        self.config = config or EnsembleConfig(mode="classification")
        self.models: dict[str, PredictiveModel] = {}
        self.target_name = "target"

    def fit(self, features: pd.DataFrame, target: pd.Series) -> VotingEnsembleModel:
        """Fit all base models."""

        if not self.model_factories:
            raise ValueError("at least one model factory is required")

        self.target_name = target.name or self.target_name
        self.models = {}
        for name, factory in self.model_factories.items():
            self.models[name] = factory().fit(features, target)
        return self

    def predict(self, features: pd.DataFrame) -> ModelPrediction:
        """Return average directional vote probability."""

        if not self.models:
            raise ValueError("ensemble must be fit before predict")

        base_predictions = {
            name: model.predict(features).predictions for name, model in self.models.items()
        }
        combined = voting_predictions(
            base_predictions,
            threshold=self.config.classification_threshold,
        )
        return ModelPrediction(
            predictions=combined,
            target_name=self.target_name,
            model_name="voting_ensemble",
        )


class StackingEnsembleModel:
    """Stack base model predictions with a simple linear meta-model."""

    def __init__(
        self,
        model_factories: Mapping[str, ModelFactory],
        config: EnsembleConfig | None = None,
    ) -> None:
        self.model_factories = dict(model_factories)
        self.config = config or EnsembleConfig()
        self.models: dict[str, PredictiveModel] = {}
        self.meta_model: object | None = None
        self.target_name = "target"
        self.base_model_names: list[str] = []

    def fit(self, features: pd.DataFrame, target: pd.Series) -> StackingEnsembleModel:
        """Fit base models and a meta-model using a chronological meta-validation fold."""

        if not self.model_factories:
            raise ValueError("at least one model factory is required")

        self.target_name = target.name or self.target_name
        split = temporal_train_test_split(
            features,
            target,
            test_size=self.config.meta_test_size,
            min_train_size=2,
        )

        meta_predictions: dict[str, pd.Series] = {}
        for name, factory in self.model_factories.items():
            model = factory().fit(split.x_train, split.y_train)
            meta_predictions[name] = model.predict(split.x_test).predictions

        meta_x = align_prediction_frame(meta_predictions)
        meta_y = split.y_test.loc[meta_x.index]
        self.meta_model = self._fit_meta_model(meta_x, meta_y)

        self.models = {}
        for name, factory in self.model_factories.items():
            self.models[name] = factory().fit(features, target)
        self.base_model_names = list(meta_x.columns)
        return self

    def predict(self, features: pd.DataFrame) -> ModelPrediction:
        """Predict by passing base-model outputs through the trained meta-model."""

        if not self.models or self.meta_model is None:
            raise ValueError("stacking ensemble must be fit before predict")

        base_predictions = {
            name: model.predict(features).predictions for name, model in self.models.items()
        }
        meta_x = align_prediction_frame(base_predictions).loc[:, self.base_model_names]
        raw = self.meta_model.predict(meta_x)
        predictions = pd.Series(raw, index=meta_x.index, name="stacking_ensemble_prediction")
        return ModelPrediction(
            predictions=predictions,
            target_name=self.target_name,
            model_name="stacking_ensemble",
        )

    def _fit_meta_model(self, meta_x: pd.DataFrame, meta_y: pd.Series) -> object:
        """Fit the linear meta-model for regression or classification."""

        if self.config.mode == "classification":
            from sklearn.linear_model import LogisticRegression

            model = LogisticRegression(max_iter=1_000)
            model.fit(meta_x, (meta_y > self.config.classification_threshold).astype(int))
            return _ProbabilityWrapper(model)

        from sklearn.linear_model import Ridge

        model = Ridge(alpha=1.0)
        model.fit(meta_x, meta_y)
        return model


class _ProbabilityWrapper:
    """Expose classifier probabilities through a regression-like predict method."""

    def __init__(self, classifier: object) -> None:
        self.classifier = classifier

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Return positive-class probabilities."""

        return self.classifier.predict_proba(features)[:, 1]
